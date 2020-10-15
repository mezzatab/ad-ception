#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:44:55 2019

@author: may706
"""

testSize=120
stage=15
def VarMethod(stage, testSize):
    GoRNN_NRMSE=[]
    GoRNN_RMSE=[]
    (T,)=GoRNNFrames[stage-1].Forecast.shape#.mean()
    Forecast=GoRNNFrames[stage-1].Forecast
    Observation=GoRNNFrames[stage-1].Observation
    nSections=int(T/testSize)
    meano=Observation[:testSize].mean()
    for i in range(1,nSections):
        GoRNN_RMSE.append(RMSE(Observation[(i)*testSize:(i+1)*testSize].tolist(),Forecast[(i)*testSize:(i+1)*testSize].tolist(),testSize))
        GoRNN_NRMSE.append(NRMSE(Observation[(i)*testSize:(i+1)*testSize].tolist(),Forecast[(i)*testSize:(i+1)*testSize].tolist(),testSize,meano))
        meano=Observation[(i)*testSize:(i+1)*testSize].mean()
    
    testSize=120
    stage=15
    GoMLP_NRMSE=[]
    GoMLP_RMSE=[]
    (T,)=GoMLPFrames[stage-1].Forecast.shape#.mean()
    Forecast=GoMLPFrames[stage-1].Forecast
    Observation=GoMLPFrames[stage-1].Observation
    nSections=int(T/testSize)
    meano=Observation[:testSize].mean()
    for i in range(1,nSections):
        GoMLP_RMSE.append(RMSE(Observation[(i)*testSize:(i+1)*testSize].tolist(),Forecast[(i)*testSize:(i+1)*testSize].tolist(),testSize))
        GoMLP_NRMSE.append(NRMSE(Observation[(i)*testSize:(i+1)*testSize].tolist(),Forecast[(i)*testSize:(i+1)*testSize].tolist(),testSize,meano))
        meano=Observation[(i)*testSize:(i+1)*testSize].mean()    
    
    testSize=120
    stage=15
    IterGoMLP_NRMSE=[]
    IterGoMLP_RMSE=[]
    (T,)=IterGoMLPFrames[stage-1].Forecast.shape#.mean()
    Forecast=IterGoMLPFrames[stage-1].Forecast
    Observation=IterGoMLPFrames[stage-1].Observation
    nSections=int(T/testSize)
    meano=Observation[:testSize].mean()
    for i in range(1,nSections):
        IterGoMLP_RMSE.append(RMSE(Observation[(i)*testSize:(i+1)*testSize].tolist(),Forecast[(i)*testSize:(i+1)*testSize].tolist(),testSize))
        IterGoMLP_NRMSE.append(NRMSE(Observation[(i)*testSize:(i+1)*testSize].tolist(),Forecast[(i)*testSize:(i+1)*testSize].tolist(),testSize,meano))
        meano=Observation[(i)*testSize:(i+1)*testSize].mean()    
    return [(GoRNN_RMSE,GoRNN_NRMSE),(GoMLP_RMSE,GoMLP_NRMSE),(IterGoMLP_RMSE,IterGoMLP_NRMSE)]