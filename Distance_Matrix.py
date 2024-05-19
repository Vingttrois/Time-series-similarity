# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:49:33 2021

@author: FanW
"""
import numpy as np

import numba as nb
from numba import cuda
import math


def DisMatrix(ArrayIn, Method):
    Sample_Num = len(ArrayIn)
    if Method == 'Euclidean':
        #欧氏距离    
        Dis_temp = np.zeros([Sample_Num,Sample_Num])
        for i_temp in range(0, Sample_Num):
            for j_temp in range(i_temp, Sample_Num):
                Dis_temp[i_temp][j_temp] = Euclidean(ArrayIn[i_temp], ArrayIn[j_temp])
        for i_temp in range(0, Sample_Num):
            for j_temp in range(i_temp, Sample_Num):
                Dis_temp[j_temp][i_temp] = Dis_temp[i_temp][j_temp]
    elif Method == 'Euclidean_UE':
        #不等长欧氏距离    
        Dis_temp = np.zeros([Sample_Num,Sample_Num])
        for i_temp in range(0, Sample_Num):
            for j_temp in range(i_temp, Sample_Num):
                Dis_temp[i_temp][j_temp] = Euclidean_UE(ArrayIn[i_temp], ArrayIn[j_temp])
        for i_temp in range(0, Sample_Num):
            for j_temp in range(i_temp, Sample_Num):
                Dis_temp[j_temp][i_temp] = Dis_temp[i_temp][j_temp] 
    elif Method == 'DTW':
        #DTW距离   
        Dis_temp = np.zeros([Sample_Num,Sample_Num])
        for i_temp in range(0, Sample_Num):
            for j_temp in range(i_temp, Sample_Num):
                Dis_temp[i_temp][j_temp] = DTW(ArrayIn[i_temp], ArrayIn[j_temp])
        for i_temp in range(0, Sample_Num):
            for j_temp in range(i_temp, Sample_Num):
                Dis_temp[j_temp][i_temp] = Dis_temp[i_temp][j_temp] 
    elif Method == 'DTW_GPU':
        #DTW距离numba cuda
        Dis_temp = DTW_GPU(ArrayIn)
    elif Method == 'Euclidean_GPU':
        #欧氏距离numba cuda
        Dis_temp = Euclidean_GPU
    return Dis_temp

def Euclidean_GPU(arrayIn):
    #生成规范矩阵
    arrayIn_x = len(arrayIn) 
    lenArr = np.zeros(arrayIn_x) 
    for i in range(0, arrayIn_x):
        lenArr[i] = len(arrayIn[i])
    arrayIn_y = 1
    arrayIn_z = int(max(lenArr))
    ArrayInput = np.zeros([arrayIn_x, arrayIn_y, arrayIn_z])
    for i in range(arrayIn_x):
        for j in range(0, int(lenArr[i])):
            ArrayInput[i][0][j] = arrayIn[i][j]
    #生成记录距离的矩阵
    SampleLen = len(arrayIn)
    Dis_temp = np.zeros([SampleLen, SampleLen])
    #分配内存
    dArrayInput = cuda.to_device(ArrayInput)
    dDis_temp = cuda.to_device(Dis_temp)
    dlenArr = cuda.to_device(lenArr)

    threadsperblock_2D = (16, 16)    
    blockspergrid_2D_x = math.ceil(arrayIn_x/ threadsperblock_2D[0])
    blockspergrid_2D_y = math.ceil(arrayIn_x/ threadsperblock_2D[1])
    blockspergrid_2D = (blockspergrid_2D_x, blockspergrid_2D_y)

    CDIS_numba_cuda_EUC[blockspergrid_2D, threadsperblock_2D](dArrayInput, dDis_temp, dlenArr)
  
    arrayB = dDis_temp.copy_to_host()
    cuda.synchronize()

    return arrayB

@cuda.jit
def CDIS_numba_cuda_EUC(dArrayInput, dDis_temp, dlenArr):  
    #dDis_temp为第一个距离矩阵,求dDis_temp
    i, j = cuda.grid(2)
    if i < dDis_temp.shape[0] and j < dDis_temp.shape[1]:
        if i == j:
            dDis_temp[i,j] = 0
        else:
            dDis_DTW = cuda.local.array(shape=(200, 200), dtype = nb.float32)
            dDis_DTW[0, 1:] = np.inf
            dDis_DTW[1:, 0] = np.inf
            dDis_DTW2 = cuda.local.array(shape=(200, 200), dtype = nb.float32)
            dDis_DTW2[0, 1:] = np.inf
            dDis_DTW2[1:, 0] = np.inf
            #---------- 
            dDis_temp[i, j] = Dist_numba_cuda_EUC(dArrayInput, dDis_DTW, dDis_DTW2, i, j, dlenArr)
    
#欧氏距离
@cuda.jit(device='gpu')
def Dist_numba_cuda_EUC(dArrayInput, dDis_DTW, dDis_DTW2, i, j, dlenArr):
    #计算每一个矩阵元素的值   
    for ki in range(1, len(dArrayInput[i][0])+1):
        for kj in range(1, len(dArrayInput[j][0])+1):
            dDis_DTW[ki, kj] = (dArrayInput[i][0][ki-1] - dArrayInput[j][0][kj-1])**2
    dDis_DTW2 = dDis_DTW
    for kki in range(1, len(dArrayInput[i][0])+1):
        for kkj in range(1, len(dArrayInput[j][0])+1):
            dDis_DTW2[kki, kkj] = dDis_DTW2[kki, kkj] + dDis_DTW[kki-1, kkj-1]
    return dDis_DTW2[int(dlenArr[i]), int(dlenArr[j])]

#DTW_GPU
def DTW_GPU(arrayIn):
    #生成规范矩阵
    arrayIn_x = len(arrayIn) #序列个数
    lenArr = np.zeros(arrayIn_x) #记录每个序列长度
    for i in range(0, arrayIn_x):
        lenArr[i] = len(arrayIn[i])
    arrayIn_y = 1
    arrayIn_z = int(max(lenArr))
    ArrayInput = np.zeros([arrayIn_x, arrayIn_y, arrayIn_z])
    for i in range(arrayIn_x):
        for j in range(0, int(lenArr[i])):
            ArrayInput[i][0][j] = arrayIn[i][j]  
    #生成记录DTW距离的矩阵
    SampleLen = len(arrayIn)
    Dis_temp = np.zeros([SampleLen, SampleLen])
    #分配内存
    dArrayInput = cuda.to_device(ArrayInput)
    dDis_temp = cuda.to_device(Dis_temp)
    dlenArr = cuda.to_device(lenArr)    
    #2D
    threadsperblock_2D = (32, 32)
    blockspergrid_2D_x = math.ceil(arrayIn_x/ threadsperblock_2D[0])
    blockspergrid_2D_y = math.ceil(arrayIn_x/ threadsperblock_2D[1])
    blockspergrid_2D = (blockspergrid_2D_x, blockspergrid_2D_y)

    CDIS_numba_cuda[blockspergrid_2D, threadsperblock_2D](dArrayInput, dDis_temp, dlenArr)
    arrayB = dDis_temp.copy_to_host()
    cuda.synchronize()
    return arrayB

@cuda.jit
def CDIS_numba_cuda(dArrayInput, dDis_temp, dlenArr):  
    i, j = cuda.grid(2)
    if i < dDis_temp.shape[0] and j < dDis_temp.shape[1]:
        if i == j:
            dDis_temp[i,j] = 0
        else:
            dDis_DTW = cuda.local.array(shape=(200, 200), dtype = nb.float32)
            dDis_DTW[0, 1:] = np.inf
            dDis_DTW[1:, 0] = np.inf
            dDis_DTW2 = cuda.local.array(shape=(200, 200), dtype = nb.float32)
            dDis_DTW2[0, 1:] = np.inf
            dDis_DTW2[1:, 0] = np.inf
            dDis_temp[i, j] = Dist_numba_cuda(dArrayInput, dDis_DTW, dDis_DTW2, i, j, dlenArr)
    
    
@cuda.jit(device='gpu')
def Dist_numba_cuda(dArrayInput, dDis_DTW, dDis_DTW2, i, j, dlenArr):  
    for ki in range(1, len(dArrayInput[i][0])+1):
        for kj in range(1, len(dArrayInput[j][0])+1):
            dDis_DTW[ki, kj] = (dArrayInput[i][0][ki-1] - dArrayInput[j][0][kj-1])**2
    dDis_DTW2 = dDis_DTW
    for kki in range(1, len(dArrayInput[i][0])+1):
        for kkj in range(1, len(dArrayInput[j][0])+1):
            dDis_DTW2[kki, kkj] = dDis_DTW2[kki, kkj] + min(dDis_DTW[kki, kkj-1],\
                     dDis_DTW[kki-1, kkj], dDis_DTW[kki-1, kkj-1])
    return math.sqrt(dDis_DTW2[int(dlenArr[i]), int(dlenArr[j])])