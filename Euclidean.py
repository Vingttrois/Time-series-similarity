# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:49:33 2021

@author: FanW
"""
import numpy as np



def Euclidean(X, Y):
    'Euclidean distance'
    arrLen = len(X)
    Distance = 0
    for i in range(0, arrLen):
        Distance = Distance + (X[i]-Y[i])**2
    return np.sqrt(Distance)

def Euclidean_UE(X, Y):
    'Euclidean distance with unequal length'
    min_len = min(len(X), len(Y))
    max_len = max(len(X), len(Y))
    temp_Euc = []
    if len(X) <= len(Y):
        short_array, long_array = X, Y
    else:
        short_array, long_array = Y, X
    for move_i in range(0, max_len-min_len+1):
        temp_Euc.append(Euclidean(short_array, long_array[move_i:move_i+min_len]))
    return np.sqrt(min(temp_Euc))