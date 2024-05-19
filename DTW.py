# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:49:33 2021

@author: FanW
"""
import numpy as np


def DTW(Arr1, Arr2):
    Len1 = len(Arr1)
    Len2 = len(Arr2)
    Lenth = max(Len1, Len2)
    #Matrix
    Dis_DTW = np.zeros([Lenth+1, Lenth+1])
    Dis_DTW[0, 1:] = np.inf
    Dis_DTW[1:, 0] = np.inf
    for ki in range(1, len(Arr1)+1):
        for kj in range(1, len(Arr2)+1):
            Dis_DTW[ki, kj] = (Arr1[ki-1] - Arr2[kj-1])**2    
    Dis_DTW2 = Dis_DTW
    for kki in range(1, len(Arr1)+1):
        for kkj in range(1, len(Arr2)+1):
            Dis_DTW2[kki, kkj] = Dis_DTW2[kki, kkj] + min(Dis_DTW[kki, kkj-1],\
                     Dis_DTW[kki-1, kkj], Dis_DTW[kki-1, kkj-1])
    return np.sqrt(Dis_DTW2[len(Arr1), len(Arr2)])