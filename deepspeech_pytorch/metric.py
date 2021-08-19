#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:07:07 2021

@author: louisbard
"""

from scipy import stats
import numpy as np


def get_correlation_values(human_answers, model_results, mode="pearson"):
    """
    :param human_answers: (list) list of human answers for each phone group (between -3 and 3 if not normalized)
    :param model_results: (list) list of delta values returned by the model (not normalized => normalized=False)
    :param mode: (str) 'pearson' or 'spearman'
    :return: Pearson or Spearman correlation value
    """
    human_answers = np.array(human_answers)
    model_results = np.array(model_results)

    # Not necessary, pearson correlation does not care if value ar normalized or not, same for spearman
    """if not normalized:
        std_human_answers = np.std(human_answers)
        std_model_results = np.std(model_results)
        human_answers /= std_human_answers
        model_results /= std_model_results"""

    if mode == "pearson":
        return stats.pearsonr(model_results, human_answers)[0]
    elif mode == "spearman":
        return stats.spearmanr(model_results, human_answers)[0]
