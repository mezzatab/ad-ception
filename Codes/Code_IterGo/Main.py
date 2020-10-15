#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Start from The beginning ## 
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

#config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':32})#device_count = {'GPU':1, 'CPU':32})
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#config.gpu_options.allow_growth = True
#keras.backend.set_session(tf.Session(config=config))


df=Preprocess.readFiles(str(ID))
#print(df)
df=Preprocess.PreprocessNR(df,list_Params)#,15)
stageEnd=df.stage_number.max()
stageOne=df.stage_number.min()

sampleSize=10
testSize=120
shortTerm=-1
#scaledFrame=Preprocess.diff_Scale(df,15)
nPredictant=5
listUncertainty=[]
scaledFrame=Preprocess.diff_Scale_midfil(df,2)
nframe=scaledFrame[['Time','prConc', 'frConc','sRate','P', 'bPropConc']].values
(tTime,z)=nframe.shape
print(tTime)    


'''
#for stage in range(15,16):    
for stage in range(stageOne,stageEnd+1):
    np.random.seed(42) 
    set_random_seed(42)        
    print(stage)        
    scaledFrame=Preprocess.diff_Scale_midfil(df,stage)
    nframe=scaledFrame[['Time','prConc', 'frConc','sRate','P', 'bPropConc']].values
    (tTime,z)=nframe.shape
    print(stage,tTime)    
    t0=sampleSize*3
    finalSection1=1+int((tTime-t0-sampleSize)/(testSize))
    finalSection2=1+int((tTime-t0+sampleSize-testSize)/(testSize))
    nSections=min(finalSection1,finalSection2)    
    uncertain=Uncertainty(nframe, sampleSize, testSize, shortTerm, nPredictant, nSections)
    listUncertainty.append(uncertain)
    listUncertainty[-1].ListRun()
    listUncertainty[-1].uncertaintyAnalysis()
    listUncertainty[-1].writeOutPut('IterativeRNN_test_ID'+str(ID)+'_stage'+str(stage)+'_5Real_s'+str(sampleSize)+'_t'+str(testSize))                
'''
