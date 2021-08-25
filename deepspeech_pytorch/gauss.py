#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:29:44 2021
@author: louisbard
"""


from scipy.stats import norm
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


def gaussrep(seq):
    # print(seq.size()
    lenght = seq.size()[0]  # time
    m = (lenght - 1) / 2
    var = (1 / 6) * (lenght - 1)
    gauss = torch.tensor(
        [norm.pdf(x, m, var) for x in range(lenght)], dtype=torch.float64
    )
    # print(gauss.size())
    gauss = gauss.type_as(seq)
    res = gauss @ (seq)

    return res


def distcos(a, b):
    """Differentiable cosine similarity"""
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    res = 1 - cos(a, b)
    return res


def distcosN(a, b):
    """Non differentiable cosine similarity nn"""
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    res = 1 - cos(a, b)

    return res.numpy()
