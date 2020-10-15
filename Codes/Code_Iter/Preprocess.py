#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.signal import medfilt
filter_size = 59
#x1 = medfilt(df.prop_conc_1,filter_size)

def readFiles(ID):
    Name='../Data/full'+ID+'.csv'
    return pd.read_csv(Name)

def Preprocess(df,List):
    df=df[List]

    df = df[(df.slurry_rate > 0 )]
    df = df[(df.prop_conc_1 > 0)]    
    df['date_time']=pd.to_datetime(df.date_time.iloc[:])#-df.date_time.iloc[0])#.dt.total_seconds()
    t = list((df.date_time-df.date_time.iloc[0]).dt.total_seconds())
    df['time'] = list(t)   
    return df 

def PreprocessNR(df,List): ## Without removing any data 
    df=df[List]

#    df = df[(df.slurry_rate > 0 )]
#    df = df[(df.prop_conc_1 > 0)]    
    df['date_time']=pd.to_datetime(df.date_time.iloc[:])#-df.date_time.iloc[0])#.dt.total_seconds()
    t = list((df.date_time-df.date_time.iloc[0]).dt.total_seconds())
    df['time'] = list(t)   
    return df 


def diff_Scale(df,stage_num):
    df1=df[(df.stage_number==stage_num)].copy()
    
    prConc=list(df1["prop_conc_1"])
    
    #### median filter 
#    x1 = medfilt(df.prop_conc_1,filter_size)        
    ##list(minmax_scale(df1[["prop_conc_1"]],feature_range=(0,1)))
#    print(prConc)
#    prConc=[val[0] for val in prConc]
    dPrConc=[0]
    for i in range(1,len(prConc)):
        dPrConc.append(prConc[i]-prConc[i-1])

        
    P=list(df1["wellhead_pressure_1"])
    #list(minmax_scale(df1[["wellhead_pressure_1"]],feature_range=(0,1)))
#    P=[val[0] for val in P]
    
    dP=[0]
    for i in range(1,len(P)):
        dP.append(P[i]-P[i-1])
        
    frConc=list(df1["fr_conc_2"])
    #list(minmax_scale(df1[["fr_conc_2"]],feature_range=(0,1)))
#    frConc=[val[0] for val in frConc]

    dFrConc=[0]
    for i in range(1,len(frConc)):
        dFrConc.append(frConc[i]-frConc[i-1])

    
    
#    sRate=list(minmax_scale(df1[['slurry_rate']],feature_range=(0,1)))
    sRate=list(df1['slurry_rate'])
    bPrConc=list(df1['bh_prop_conc'])
    dBprConc=[0]
    for i in range(1,len(bPrConc)):
        dBprConc.append(bPrConc[i]-bPrConc[i-1])
    
    Z=df1[['time']].values
#    print(Z[0,0])
    time=[t[0]-Z[0,0] for t in df1[['time']].values]
    return pd.DataFrame({'Time':time,'P':P,'dP':dP, 'frConc':frConc, 'Diff_FracConc':dFrConc,'sRate':sRate,'prConc':prConc, 'Diff_prConc':dPrConc ,'bPropConc':bPrConc ,'Diff_bPropConc':dBprConc})

def diff_Scale_midfil(df,stage_num):
    filter_size = 59
    df1=df[(df.stage_number==stage_num)].copy()
    
#    prConc=list(df1["prop_conc_1"])
    
    #### median filter 
    prConc = medfilt(df1.prop_conc_1,filter_size)        
    ##list(minmax_scale(df1[["prop_conc_1"]],feature_range=(0,1)))
#    print(prConc)
#    prConc=[val[0] for val in prConc]
    dPrConc=[0]
    for i in range(1,len(prConc)):
        dPrConc.append(prConc[i]-prConc[i-1])

    #### median filter 
    P = medfilt(df1.wellhead_pressure_1,filter_size)        
      #  
#    P=list(df1["wellhead_pressure_1"])
    #list(minmax_scale(df1[["wellhead_pressure_1"]],feature_range=(0,1)))
#    P=[val[0] for val in P]
    
    dP=[0]
    for i in range(1,len(P)):
        dP.append(P[i]-P[i-1])
    #### median filter 
    frConc = medfilt(df1.fr_conc_2,filter_size)        
        
#    frConc=list(df1["fr_conc_2"])
    #list(minmax_scale(df1[["fr_conc_2"]],feature_range=(0,1)))
#    frConc=[val[0] for val in frConc]

    dFrConc=[0]
    for i in range(1,len(frConc)):
        dFrConc.append(frConc[i]-frConc[i-1])

    
    
#    sRate=list(minmax_scale(df1[['slurry_rate']],feature_range=(0,1)))
    #### median filter 
    sRate = medfilt(df1.slurry_rate,filter_size)        
    bPrConc = medfilt(df1.bh_prop_conc,filter_size)        
        
        
#    sRate=list(df1['slurry_rate'])
#    bPrConc=list(df1['bh_prop_conc'])
    dBprConc=[0]
    for i in range(1,len(bPrConc)):
        dBprConc.append(bPrConc[i]-bPrConc[i-1])
    
    Z=df1[['time']].values
#    print(Z[0,0])
    time=[t[0]-Z[0,0] for t in df1[['time']].values]
    return pd.DataFrame({'Time':time,'P':P,'dP':dP, 'frConc':frConc, 'Diff_FracConc':dFrConc,'sRate':sRate,'prConc':prConc, 'Diff_prConc':dPrConc ,'bPropConc':bPrConc ,'Diff_bPropConc':dBprConc})

