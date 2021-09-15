#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:10:37 2021

@author: louisbard
"""

# importation of modules

import pandas as pd
from deepspeech_pytorch.real_dtw import compute_dtw
import numpy as np
import torch.nn as nn
from deepspeech_pytorch.gauss import distcos
from deepspeech_pytorch.gauss import distcosN


def get_res(TGT, OTH, X):

    """Compute delta value for the representation obtained"""
    # Since we are now out of the model, we do not need the differentiable version of this
    # So I included the real dtw
    # sdtw = SoftDTW(gamma=1.0, normalize=True, dist='cosine')
    # and so here we have OTH,X minus TGT,X

    if TGT.shape == 0 or OTH.shape == 0 or TGT.shape == 0:
        print("Problem size 0 encountered")
        return 0.0

    OTH_X = compute_dtw(OTH, X, dist_for_cdist="cosine", norm_div=True)
    TGT_X = compute_dtw(TGT, X, dist_for_cdist="cosine", norm_div=True)

    if np.isnan(OTH_X) or np.isnan(TGT_X):
        return 0.0
    if np.isinf(OTH_X) or np.isinf(TGT_X):
        return 0.0

    delta = OTH_X - TGT_X

    if np.isinf(delta):
        return 0.0

    return delta


def get_res_gauss(TGT, OTH, X):

    """Compute delta value for the representation obtained"""
    # Since we are now out of the model, we do not need the differentiable version of this
    # So I included the real dtw
    # sdtw = SoftDTW(gamma=1.0, normalize=True, dist='cosine')
    # and so here we have OTH,X minus TGT,X

    if TGT.shape == 0 or OTH.shape == 0 or TGT.shape == 0:
        print("Problem size 0 encountered")
        return 0.0

    OTH_X = distcosN(OTH, X)
    TGT_X = distcosN(TGT, X)
    if np.isnan(OTH_X) or np.isnan(TGT_X):
        return 0.0
    if np.isinf(OTH_X) or np.isinf(TGT_X):
        return 0.0

    delta = OTH_X - TGT_X

    if np.isinf(delta):
        return 0.0

    return delta


def complet_csv(human_csv, delta, triplet_id):

    """Complete the human csv file with the new delta values"""

    df = pd.read_csv(human_csv)
    df = df[
        [
            "subject_id",
            "triplet_id",
            "language_TGT",
            "language_OTH",
            "phone_TGT",
            "phone_OTH",
            "context",
            "user_ans",
            "dataset"
        ]
    ]

    # We average for triplet
    gf = df.groupby(
        [
            "triplet_id",
            "language_TGT",
            "language_OTH",
            "phone_TGT",
            "phone_OTH",
            "context",
            "dataset"
        ],
        as_index=False,
    )
    ans = gf.user_ans.mean()

    # we divide user ans by 3 for specific datasets
    ans[ans['dataset'] == "WorldVowels", ['user_ans']] = ans[ans['dataset'] == "WorldVowels", ['user_ans']]/3.
    ans[ans['dataset'] == "zerospeech", ['user_ans']] = ans[ans['dataset'] == "zerospeech", ['user_ans']] / 3.

    # We complete with delta values
    df2 = ans[ans["triplet_id"].isin(triplet_id)].copy()
    df2["delta_values"] = 0

    count = 0
    for id in triplet_id:
        df2.loc[df2["triplet_id"] == id, ["delta_values"]] = delta[count]
        count += 1

    # Then we average for context
    gf = df2.groupby(
        ["language_TGT", "language_OTH", "phone_TGT", "phone_OTH", "context"],
        as_index=False,
    )
    ans = gf.user_ans.mean()
    val_delta = gf.delta_values.mean()
    ans["delta_values"] = val_delta["delta_values"]

    # then we average over phone contrast
    gf = ans.groupby(
        ["language_TGT", "language_OTH", "phone_TGT", "phone_OTH"], as_index=False
    )
    ans = gf.user_ans.mean()
    val_delta = gf.delta_values.mean()
    ans["delta_values"] = val_delta["delta_values"]

    # the we average over order or the other way around (contrast o/a or a/o need to be considered as the same)
    res = ans.copy()
    res["phone_TGT"] = ans["phone_OTH"]
    res["phone_OTH"] = ans["phone_TGT"]

    total = pd.concat(
        [ans, res],
        axis=0,
    )
    # print('TOTAL#####', total)
    gf = total.groupby(
        ["language_TGT", "language_OTH", "phone_TGT", "phone_OTH"], as_index=False
    )
    ans = gf.user_ans.mean()
    val_delta = gf.delta_values.mean()
    ans["delta_values"] = val_delta["delta_values"]

    return ans
