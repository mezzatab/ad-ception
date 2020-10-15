#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:34:04 2019

@author: may706
"""
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
import WellHeadClass #import WellheadPressurePrediction 
#import WellHeadVal 
#np.random.seed(42)
#set_random_seed(42)
############################  Class Uncertainty for list of predictors ####################        
        
class Uncertainty:

    def __init__(self, df, sampleSize, testSize, shortTerm, nPredict, nSections=10):
        self.Preds=[]
        (self.tTime,xAxis)=df.shape
        self.nPredict=nPredict
        self.sampleSize=sampleSize
        self.testSize=testSize
        self.shortTerm=shortTerm
        self.nSections=nSections
#        self.nSections=1+int((self.tTime-self.sampleSize)/(self.testSize))
        for i in range(nPredict):
            self.Preds.append(WellHeadClass.WellheadPressurePrediction(df, self.sampleSize, self.testSize, self.shortTerm, 'RNN'))
            
        self.METRICS = {
            "NMSE":self.NMSE, # NSE 
            "NRMSE":self.NRMSE, # NSE             
            "RMSE":self.rmean_squared_error,
            "WB":self.WB,
            "SMAPE":self.smape,
            "MAPE":self.mape
        }
        
        self.metricsGrowth = {
            "NMSE":[], # NSE 
            "NRMSE":[], # NSE             
            "RMSE":[],
            "WB":[],
            "SMAPE":[],
            "MAPE":[]            
        }
        
        self.bestMetricsGrowth = {
            "NMSE":[], # NSE 
            "NRMSE":[], # NSE             
            "RMSE":[],
            "WB":[],
            "SMAPE":[],
            "MAPE":[]            
        }
        
        
        self.worstMetricsGrowth = {
            "NMSE":[], # NSE 
            "NRMSE":[], # NSE             
            "RMSE":[],
            "WB":[],
            "SMAPE":[],
            "MAPE":[]            
        }
        
    def metric(self,X,Y):
        for key in self.METRICS.keys():
            self.metricsGrowth[key].append(self.METRICS[key](X,Y))#,self.testSize))
        
    def bestMetric(self,X,Y):  ## Refer to best prediction ## 
        for key in self.METRICS.keys():
            self.bestMetricsGrowth[key].append(self.METRICS[key](X,Y))#,self.testSize))

    def worstMetric(self,X,Y):  ## Refer to worst prediction ## 
        for key in self.METRICS.keys():
            self.worstMetricsGrowth[key].append(self.METRICS[key](X,Y))#,self.testSize))
            
            
    def ListRun(self):
        for i in range(self.nPredict):
            print(i)
            self.Run(self.Preds[i])
        self.compTime=[]
        L=len(self.Preds[0].compTime)
        for i in range(L):
            avTime=0
            for j in range(self.nPredict):
                avTime=avTime+self.Preds[j].compTime[i]
            avTime=avTime/self.nPredict
            self.compTime.append(avTime)

            
    def Run(self, WPP): 
        totalTime=0
        t0=3*self.sampleSize
        for i in range(self.nSections):
            print(i)
            t1=time.time()
            WPP.train(t0+i*self.testSize)
            t2=time.time()
            dt=t2-t1
            totalTime=totalTime+dt
            WPP.compTime.append(totalTime)    
            WPP.predict()   
            
    def uncertaintyAnalysis(self):
        self.cTime=self.Preds[0].cTime
        self.cY=self.Preds[0].cY_test
        (x,)=self.Preds[0].cYp_test.shape[:]
        self.SigmaCalc=np.zeros(x)
        self.avgCalc=np.zeros(x)  
        for j in range(x):        
            avg=0            
            for i in range(self.nPredict):
                avg=avg+self.Preds[i].cYp_test[j]
            avg=avg/self.nPredict
            self.avgCalc[j]=avg                          
        for j in range(x):
            s2=0
            for i in range(self.nPredict):
                s2=s2+(self.Preds[i].cYp_test[j]-self.avgCalc[j])**2
            s2=s2/self.nPredict
            self.SigmaCalc[j]=math.sqrt(s2)
            
        self.compTime=[]
        L=len(self.Preds[0].compTime)
        for i in range(L):
            avTime=0
            for j in range(self.nPredict):
                avTime=avTime+self.Preds[j].compTime[i]
            avTime=avTime/self.nPredict
            self.compTime.append(avTime)
        self.bestPredictor()            
            
        for i in range(self.nSections-1):
            self.metric(self.cY[i*self.testSize:(i+1)*self.testSize],self.avgCalc[i*self.testSize:(i+1)*self.testSize])
            self.bestMetric(self.cY[i*self.testSize:(i+1)*self.testSize],self.Preds[self.bestPred].cYp_test[i*self.testSize:(i+1)*self.testSize])
            self.worstMetric(self.cY[i*self.testSize:(i+1)*self.testSize],self.Preds[self.worstPred].cYp_test[i*self.testSize:(i+1)*self.testSize])
            
        i=self.nSections-1
        self.metric(self.cY[i*self.testSize:i*self.testSize+self.Preds[0].rT],self.avgCalc[i*self.testSize:i*self.testSize+self.Preds[0].rT])
        self.bestMetric(self.cY[i*self.testSize:i*self.testSize+self.Preds[0].rT],self.Preds[self.bestPred].cYp_test[i*self.testSize:i*self.testSize+self.Preds[self.bestPred].rT])
        self.worstMetric(self.cY[i*self.testSize:i*self.testSize+self.Preds[0].rT],self.Preds[self.worstPred].cYp_test[i*self.testSize:i*self.testSize+self.Preds[self.worstPred].rT])

    def bestPredictor(self):
        i=self.nSections-1
        self.rmse=[]
        self.bestPred=0
        self.worstPred=0
        rmBest=self.METRICS['RMSE'](self.cY[:i*self.testSize+self.Preds[0].rT],self.Preds[0].cYp_test[:i*self.testSize+self.Preds[0].rT])
        rmWorst=rmBest
        for j in range(self.nPredict):
            rm=self.METRICS['RMSE'](self.cY[:i*self.testSize+self.Preds[j].rT],self.Preds[j].cYp_test[:i*self.testSize+self.Preds[j].rT])
            if rm < rmBest:
                self.bestPred=j
                rmBest=rm
            if rm > rmWorst:
                self.worstPred=j
                rmWorst=rm                
            self.rmse.append(rm)
#        self.metric(self.cY[i*self.testSize:i*self.testSize+self.Preds[0].rT],self.avgCalc[i*self.testSize:i*self.testSize+self.Preds[0].rT])
        
            
    def WB(self, Yo,Yp):
        (L,)=Yo.shape
        sum_o=0
        sum_p=0
        for i in range(L):
            sum_o+=Yo[i]
            sum_p+=Yp[i]
        return 1-abs(1-sum_p/sum_o)
    
    def smape(self, Yo,Yp):
        (L,)=Yo.shape        
        Sum=0
        for i in range(L):
            Sum=Sum+2*abs(Yp[i]-Yo[i])/(abs(Yp[i])+abs(Yo[i]))
        return Sum/L
    
    def mape(self, Yo, Yp):
        (L,)=Yo.shape        
        Sum=0
        for i in range(L):
            Sum=Sum+abs(Yp[i]-Yo[i])/(abs(Yo[i]))
        return Sum/L
    
    
    def NMSE(self, Yo, Yp):
        (L,)=Yo.shape        
        sum_o=0
        sum_p=0
        for i in range(L):
            sum_o+=Yo[i]
            sum_p+=Yp[i]
        meano=sum_o/L
        meanp=sum_p/L
    
        numerator=0
        denumerator=0
        for i in range(len(Yo)):
            numerator=numerator+(Yp[i]-Yo[i])**2
            denumerator=denumerator+(Yo[i]-meano)**2
        return 1-numerator/denumerator

    
    def NRMSE(self, Yo, Yp):
        (L,)=Yo.shape
        sum_o=0
        sum_p=0
        for i in range(L):
            sum_o+=Yo[i]
            sum_p+=Yp[i]
        meano=sum_o/L
        meanp=sum_p/L
    
        numerator=0
        denumerator=0
        for i in range(len(Yo)):
            numerator=numerator+abs(Yp[i]-Yo[i])
            denumerator=denumerator+abs(Yo[i]-meano)
        return 1-numerator/denumerator
    
    
    def rmean_squared_error(self, Yo, Yp):
        Sum=0
        (L,)=Yo.shape        
        for i in range(L):
            Sum=Sum+(Yo[i]-Yp[i])**2
        return math.sqrt(Sum/L)

            
            
    def writeOutPut(self,file):
        self.Dict={'Time':self.cTime,'Observation':self.cY,'Forecast':self.avgCalc,'ConfidenceLevel':2*self.SigmaCalc,'BestForecast':self.Preds[self.bestPred].cYp_test,'WorstForecast':self.Preds[self.worstPred].cYp_test}
        self.ForecastDict={'Forecast_'+str(i+1):self.Preds[i].cYp_test for i in range(self.nPredict)}
        self.Dict.update(self.ForecastDict)
        self.DataFrame=pd.DataFrame(self.Dict)#{'Time':self.cTime,'Observation':self.cY,'Forecast':self.avgCalc,'ConfidenceLevel':2*self.SigmaCalc})#,ForecastDict})
        self.DataFrame.to_csv("/home/may706/LinuxBox/MetaData/Uncertainty-Forecast_"+file+".csv")
        
        self.Metric={'Section':[self.Preds[0].cTime[self.testSize*i] for i in range(self.nSections)], 'NMSE':self.metricsGrowth['NMSE'], 'NRMSE':self.metricsGrowth['NRMSE'],'MAPE':self.metricsGrowth['MAPE'], 'SMAPE':self.metricsGrowth['SMAPE'],'RMSE':self.metricsGrowth['RMSE'],'CalcTime':self.compTime}        
        self.BestMetric={'Section':[self.Preds[self.bestPred].cTime[self.testSize*i] for i in range(self.nSections)], 'Best_NMSE':self.bestMetricsGrowth['NMSE'], 'Best_NRMSE':self.bestMetricsGrowth['NRMSE'],'Best_MAPE':self.bestMetricsGrowth['MAPE'], 'Best_SMAPE':self.bestMetricsGrowth['SMAPE'],'Best_RMSE':self.bestMetricsGrowth['RMSE']}        
        self.WorstMetric={'Section':[self.Preds[self.worstPred].cTime[self.testSize*i] for i in range(self.nSections)], 'Worst_NMSE':self.worstMetricsGrowth['NMSE'], 'Worst_NRMSE':self.worstMetricsGrowth['NRMSE'],'Worst_MAPE':self.worstMetricsGrowth['MAPE'], 'Worst_SMAPE':self.worstMetricsGrowth['SMAPE'],'Worst_RMSE':self.worstMetricsGrowth['RMSE']}                
        self.Metric.update(self.BestMetric)
        self.Metric.update(self.WorstMetric)
        self.MetricFrame=pd.DataFrame(self.Metric)        
                
        self.MetricFrame.to_csv("/home/may706/LinuxBox/MetaData/MetricFrame_"+file+".csv")
