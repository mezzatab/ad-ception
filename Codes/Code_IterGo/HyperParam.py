## Start from The beginning ## 
import setup 
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
#from WellHeadClass import WellheadPressurePrediction

np.random.seed(42) 
set_random_seed(42)




df=Preprocess.readFiles("4230133468")
#print(df)
df=Preprocess.Preprocess(df,list_Params)#,15)
#scaledFrame=Preprocess.diff_Scale(df,15)
scaledFrame=Preprocess.diff_Scale_midfil(df,15)
#scaledFrame.P=scaledFrame.P-scaledFrame.P.iloc[0]
nframe=scaledFrame[['Time','prConc', 'frConc','sRate','P', 'bPropConc']].values

config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':32})#device_count = {'GPU':1, 'CPU':32})
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

## Metric frames ## 


#config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':32})
#tf.config.gpu.set_per_process_memory_fraction(0.4)
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)
#print(device_lib.list_local_devices())

                                      ####### Int main ####### 

## COnsider dP ## 
#testSize=120 
#listSampleSize=[10, 20, 30, 60, 120, 300] 
listSampleSize=[5,10,20,30, 60]#5, 10, 20, 30, 60, 120] #, 180, 240, 300]
listTestSize=[120]#[300, 240, 120, 60]#, 30]    #5, 10, 20, 30, 60, 120, 180, 240, 300] 
#listSampleSize=[60, 120, 180, 240, 300] 
#testSize=[120 for i in range(len(listSampleSize))]
nPredictant=5
shortTerm=-1 ## Negative means learn from beginning for each training session 
listUncertainty=[]
#BestRMSE=[]
#BestNRMSE=[] 
for testSize in listTestSize:
    for sampleSize in listSampleSize:
        np.random.seed(42) 
        set_random_seed(42)        
        print(testSize,sampleSize)
        (tTime,z)=nframe.shape
        t0=sampleSize*3
        finalSection1=1+int((tTime-t0-sampleSize)/(testSize))
        finalSection2=1+int((tTime-t0+sampleSize-testSize)/(testSize))
        nSections=min(finalSection1,finalSection2)    
        uncertain=Uncertainty(nframe, sampleSize, testSize, shortTerm, nPredictant, nSections)
        listUncertainty.append(uncertain)
        listUncertainty[-1].ListRun()
        listUncertainty[-1].uncertaintyAnalysis()
        listUncertainty[-1].writeOutPut('IterativeRNN_5Real_s'+str(sampleSize)+'_t'+str(testSize))
        


        
        
                
