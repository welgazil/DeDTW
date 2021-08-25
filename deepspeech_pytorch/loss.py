#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:53:32 2021

@author: louisbard
"""

# import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F
from deepspeech_pytorch.soft_dtw import SoftDTW

from deepspeech_pytorch.gauss import distcos


class DTWLosslabels(nn.Module):
    def __init__(self, representation):
        super(DTWLosslabels, self).__init__()
        self.sdtw = SoftDTW(gamma=1.0, normalize=True, dist="cosine")
        self.criterion = nn.MSELoss()
        self.representation = representation

    def forward(self, TGT, OTH, X, labels):
        TGT, OTH, X = TGT.to(torch.float32), OTH.to(torch.float32), X.to(torch.float32)
        labels = torch.as_tensor(labels[0], dtype=torch.float)
        if self.representation == "gauss":
            loss = self.criterion(distcos(OTH, X) - distcos(TGT, X), labels)
        else:
            loss = self.criterion(self.sdtw(OTH, X) - self.sdtw(TGT, X), labels)
        return loss


class DTWLosswithoutlabels(nn.Module):
    def __init__(self, representation):
        super(DTWLosswithoutlabels, self).__init__()
        self.sdtw = SoftDTW(gamma=1.0, normalize=True, dist="cosine")
        self.representation = representation

    def forward(self, TGT, OTH, X, labels):
        TGT, OTH, X = TGT.to(torch.float32), OTH.to(torch.float32), X.to(torch.float32)
        # labels = torch.as_tensor(labels[0], dtype=torch.float)
        if self.representation == "gauss":
            loss = distcos(TGT, X) - distcos(OTH, X)
        else:
            loss = self.sdtw(TGT, X) - self.sdtw(
                OTH, X
            )  # it is ok to put it this way because we want to minimize this
        # otherwise, the delta value is the other way around
        return loss
