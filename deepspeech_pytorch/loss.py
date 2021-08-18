#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:53:32 2021

@author: louisbard
"""

#import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
from soft_dtw import SoftDTW
# rajoutez des paramètres éventuellements
criterion = nn.MSELoss()

class DTWLoss(nn.Module):
    def __init__(self):
        super(DTWLoss, self).__init__()
        self.sdtw = SoftDTW(gamma=1.0,normalize=True,dist='cosine')

    def forward(self, TGT, OTH, X, labels):
        TGT,OTH,X = TGT.to(torch.float32), OTH.to(torch.float32), X.to(torch.float32)
      #  print(labels)
        labels = torch.as_tensor(labels[0],dtype=torch.float)
        
        
       # loss = self.sdtw(TGT,X) - self.sdtw(OTH,X)
        loss =  criterion(self.sdtw(TGT,X) - self.sdtw(OTH,X),labels) # it is ok to put it this way because we want to minimize this
        # otherwise, the delta value is the other way around
        return loss
