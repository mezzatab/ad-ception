#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:26:20 2019
@author: Mohammadmehdi Ezzatabadipour  
Contact: mehdi.ezatabadi3@gmail.com 

"""

## Start from The beginning ## 
## Time series from (0,T) using sliding window turn into small images with output Pressure ## 
#import setup 
import Preprocess 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
from tensorflow import keras, set_random_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  mean_squared_error
from functools import partial
from collections import defaultdict 
from pprint import pprint
from Uncertainty import Uncertainty

#np.random.seed(42) 
#set_random_seed(42)

config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':32})#device_count = {'GPU':1, 'CPU':32})
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

listNames=["4230133468","4247536931","4247537091","4247537106","l4247537154","4247537186"] ## List of wells 
list_Params=['date_time','stage_number','prop_conc_1','bh_prop_conc','fr_conc_2','slurry_rate','wellhead_pressure_1'] ## List of Params in wells s Data

ID=4230133468 ## Well ID 

            #### Reading the file for Well and preprocess including Median Filter ## 

df=Preprocess.readFiles(str(ID))
#print(df)
df=Preprocess.Preprocess(df,list_Params)#,15)

                    ###############################################
stageEnd=df.stage_number.max()
stageOne=df.stage_number.min()

sampleSize=2 ## Hyper Param n1 
testSize=120  ## Hyper Param n2=n*n1 Forecasting Window length 
shortTerm=-1 ### Negative means consider the whole data from the begining, otherwise it tells how deep in history we collect the data 
#scaledFrame=Preprocess.diff_Scale(df,15)
nPredictant=5 ## Number of forecasts to calculate uncertainty 
listUncertainty=[]
#for stage in range(1,2):    
for stage in range(stageOne,stageEnd+1): ## Do forecast for all stages 
    np.random.seed(42) 
    set_random_seed(42)        
    print(stage)        
    scaledFrame=Preprocess.diff_Scale_midfil(df,stage)
    nframe=scaledFrame[['Time', 'bPropConc', 'frConc','sRate','P', 'prConc']].values
    (tTime,z)=nframe.shape
    print(stage,tTime)    
    t0=sampleSize*3
    ### Determining the number of forecast segments every testSize ##
    finalSection1=1+int((tTime-t0-sampleSize)/(testSize))
    finalSection2=1+int((tTime-t0+sampleSize-testSize)/(testSize))
    nSections=min(finalSection1,finalSection2)    
    #####################################################
    uncertain=Uncertainty(nframe, sampleSize, testSize, shortTerm, nPredictant, nSections, engine='MLP') ## Choose pred Ebgine "MLP"
    listUncertainty.append(uncertain)
    listUncertainty[-1].ListRun() ## Run 
    listUncertainty[-1].uncertaintyAnalysis() ## Caluclate the avg and sigma 
    listUncertainty[-1].writeOutPut('OneGoMLP_test_ID'+str(ID)+'_stage'+str(stage)+'_5Real_s'+str(sampleSize)+'_t'+str(testSize))                
