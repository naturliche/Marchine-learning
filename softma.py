# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:07:30 2019

@author: natur
"""

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

scores = [1000, 1001, 1003]
print(softmax(scores))