#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:36:53 2021

@author: louisbard
"""

# Preprocessing

# takes as input the csv file of the triplets and creates the train, validation and test sets.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def create_csv(triplet_csv):
    df = pd.read_csv(triplet_csv)
    train, test_ = train_test_split(df, train_size=0.80, shuffle=True)
    split = int(np.floor(0.5 * len(test_)))
    test = test_[split:]
    valid = test_[:split]
    name = os.path.basename(triplet_csv)
    train.to_csv("train_" + name, index=False)
    test.to_csv("test_" + name, index=False)
    valid.to_csv("valid_" + name, index=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "triplet_csv", metavar="triplet_csv", help="file containing triplets audios"
    )

    args = parser.parse_args()

    create_csv(args.triplet_csv)
