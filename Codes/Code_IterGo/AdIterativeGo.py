#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Start from The beginning ## 
## Time series from (0,T) using sliding window turn into small images with output Pressure ## 
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
from scipy.signal import medfilt
filter_size = 59


#np.random.seed(42)
#set_random_seed(42)

class WellheadPressurePrediction:

    def __init__(self, df, sample_size, forecast_window, shortTerm, engine="default"):
        self.df = df
        (self.totalTime, xAxis)=df.shape
        self.nFeatures=4
        self.shortTerm=shortTerm
        self.sampleSize=sample_size
        self.testSize=sample_size
        self.fWindow=forecast_window
        self.cY_test=None
        self.cYp_test=None  
        self.cTime=None        
        self.compTime=[]
        self.engine=engine
        self.preFry=np.zeros(self.testSize)
        
        self.Engine = {
            "default":setup.setup,
            "1layer":setup.setup1layer,            
            "2layer":setup.setup2layer,                              
            "2layerRNN":setup.setup2layerRNN,
            "RNN":setup.setupRNN,
            "MLP":setup.setupMLP                                                                                      
        }
        
        self.model=self.Engine[self.engine](4, self.sampleSize, self.sampleSize, 'mse') ## sample size and test size similar 
#        print(self.model,self.engine,self.Engine[self.engine])
        #self.model=setup.setup(4, sample_size, test_size, 'mse')        
        
        self.section=0
        
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
        

        
    def metric(self):
        for key in self.METRICS.keys():
#            print(self.Y_test.shape,self.Yp_test.shape,self.rT)
            self.metricsGrowth[key].append(self.METRICS[key](self.Y_test[:,0],self.Yp_test[:],self.rT))

                                                    
    def train(self,T):## Shape input for Multilayer Perceptron is not Matrix but an array, we need flatten input for MLP
        self.T=T
        self.preprocess()
        
        if self.engine=='MLP':    
            (s1,s2,s3)=self.X.shape
            self.Xtemp=np.zeros((s1,s2*s3))
            for i in range(s1):
                self.Xtemp[i]=self.X[i].flatten()
            self.X=self.Xtemp
        if self.shortTerm < 0: ## It trains from the begininng ##  
            early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1, mode='min')
#            self.history=self.model.fit(self.X[self.sampleSize:self.T-self.sampleSize], self.Y[self.sampleSize:self.T-self.sampleSize], validation_split=0.15, epochs=100, batch_size=200, verbose=0, callbacks=[early_stop])
            self.history=self.model.fit(self.X[self.sampleSize:self.T-self.sampleSize], self.Y[self.sampleSize:self.T-self.sampleSize], epochs=100, batch_size=200, verbose=0, callbacks=[early_stop])            
        else:
            if self.T-self.shortTerm-self.sampleSize > 0:  
                early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1, mode='min')
                self.history=self.model.fit(self.X[self.T-self.shortTerm-self.sampleSize:self.T-self.sampleSize], self.Y[self.T-self.shortTerm-self.sampleSize:self.T-self.sampleSize], epochs=20, batch_size=200, verbose=0, callbacks=[early_stop])
            else:
                early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1, mode='min')
                self.history=self.model.fit(self.X[self.sampleSize:self.T-self.sampleSize], self.Y[self.sampleSize:self.T-self.sampleSize], epochs=20, batch_size=200, verbose=0, callbacks=[early_stop])
            

                                            ## Train from short period ##            
 
        
##########    Predict with no ground Truth #######################            

#### The assumption is T+sampleSize is existing in data file #### 

    def MLP(self):
        if self.engine=='MLP':
            (s1,s2,s3)=self.testX.shape
            self.Xtemp=np.zeros((s1,s2*s3))#self.testX
            for i in range(s1):
                self.Xtemp[i]=self.testX[i].flatten()
            self.testX=self.Xtemp
        
#### The assumption is T+sampleSize is existing in data file #### 
                ## sampleSize=testSize ##                 
#############################################################################################
##################   Iterative Prediction incombined with multiple step prediction ##########
            
### Idea behind this test and sample size is the same size #########
                 
    def predict(self):
        self.section=self.section+1
        
        ### If the prediction goes byond the existing data file ### rT != fWindow 
        self.rT=self.fWindow
        if self.T+self.fWindow > self.totalTime:
            self.rT=self.totalTime-self.T
        self.fTimes=int(self.rT/self.sampleSize)
        self.rT=self.fTimes*self.sampleSize
        
        self.tX=self.scalerX.transform(self.df[self.T-self.sampleSize:self.T+self.rT,:])        
        
        self.testX=np.zeros((1,self.sampleSize,4))        ## One point for prediction 
        self.Yp_test=None        
        
        self.timeTest=np.zeros((self.rT))        
        for i in range(self.rT):
            self.timeTest[i]=self.df[self.T+i,0]
            #        self.Time=self.Time.tolist()            
        #### Prediction Engine In Iterative Go Fasion ####                                             

#        self.preFry=self.tX[:self.sampleSize,4]## Predict for same sample   size ##
        self.testX[0,:,self.nFeatures-1]=self.tX[:self.sampleSize,self.nFeatures]      
        for i in range(1,self.fTimes+1):        ## fTimes=fWindow/sampleSize ##
            for j in range(self.nFeatures-1):
                self.testX[0,:,j]=self.tX[i*self.sampleSize:(i+1)*self.sampleSize,j+1]#.tolist()
            self.MLP() ## Reshape the structure for MLP setup if MLP setup is chosen 
            self.preFry=self.model.predict(self.testX) ## Predict for same sample   size ##
            self.testX=self.testX.reshape((1,self.sampleSize,4))            
            self.testX[0,:,self.nFeatures-1]=self.preFry                  
            self.Yp=self.scalerY.inverse_transform(self.preFry.reshape(-1,1))        ## For next sampleSize 
            if self.Yp_test is not None:
                self.Yp_test=np.concatenate((self.Yp_test,self.Yp),axis=None)        
            else:
                self.Yp_test=self.Yp    ## Yp is Y predict ## 
                
        ###########################        
        self.Y_test=self.df[self.T:self.T+self.rT,4].reshape(-1,1)    #### The real time data for testing ###
        self.metric()
        if self.cY_test is not None:
            self.cYp_test=np.concatenate((self.cYp_test,self.Yp_test),axis=None)
            self.cY_test=np.concatenate((self.cY_test,self.Y_test),axis=None)   
            self.cTime=np.concatenate((self.cTime,self.timeTest),axis=None)            
        else:
            self.cYp_test=self.Yp_test
            self.cY_test=self.Y_test
            self.cTime=self.timeTest#-self.tsart
#        print(self.cY_test.shape,self.cYp_test.shape)
##############################################################################################



##################   End of Iterative Prediction incombined with multiple step prediction ##########            
##########III####

    def preprocess(self): ## Scaling data for time T and using pressure as input
        ### Makse sure T < totalTime
        
        self.df_slice=self.df[:self.T,:]
        self.scalerX=MinMaxScaler(feature_range=(0, 1)) ## Scaling the input ## 
        self.scalerX=self.scalerX.fit(self.df[:self.T,:])
        self.scaledX=self.scalerX.transform(self.df[:self.T,:])
        
        self.scalerY=MinMaxScaler(feature_range=(0, 1))
        self.scalerY=self.scalerY.fit(self.df[:self.T,4].reshape(-1,1))
        self.scaledY=self.scalerY.transform(self.df[:self.T-self.sampleSize+self.testSize,4].reshape(-1,1))
        
        
        self.X=np.zeros((self.T,self.sampleSize,4))
        
        self.Y=np.zeros((self.T,self.testSize))
        for i in range(self.sampleSize,self.T-self.sampleSize): ## To set the marginal border before the totalTime  
            self.X[i,:,0]=self.scaledX[i:i+self.sampleSize,1]#.tolist()
            self.X[i,:,1]=self.scaledX[i:i+self.sampleSize,2]#.tolist()            
            self.X[i,:,2]=self.scaledX[i:i+self.sampleSize,3]#.tolist()     # #               
            self.X[i,:,3]=self.scaledX[i-self.sampleSize:i,4]#.tolist()     # # Pressure Items # # 
        for i in range(self.sampleSize,self.T-self.sampleSize):                 
            self.Y[i,:]=self.scaledY[i:i+self.testSize,0]

##########
        

    def WB(self, Yo,Yp, L):
        sum_o=0
        sum_p=0
        for i in range(L):
            sum_o+=Yo[i]
            sum_p+=Yp[i]
        return 1-abs(1-sum_p/sum_o)
    
    def smape(self, Yo,Yp, L):
        Sum=0
        for i in range(L):
            Sum=Sum+2*abs(Yp[i]-Yo[i])/(abs(Yp[i])+abs(Yo[i]))
        return Sum/L
    
    def mape(self, Yo,Yp, L):
        Sum=0
        for i in range(L):
            Sum=Sum+abs(Yp[i]-Yo[i])/(abs(Yo[i]))
        return Sum/L
    

    def NMSE(self, Yo, Yp, L):
        sum_o=0
        sum_p=0
        for i in range(L):
            sum_o+=Yo[i]
            sum_p+=Yp[i]
        meano=sum_o/L
        meanp=sum_p/L
    
        numerator=0
        denumerator=0
        for i in range(L):
            numerator=numerator+(Yp[i]-Yo[i])**2
            denumerator=denumerator+(Yo[i]-meano)**2
        return 1-numerator/denumerator
    
    
    def NRMSE(self, Yo, Yp, L):
        sum_o=0
        sum_p=0
        for i in range(L):
            sum_o+=Yo[i]
            sum_p+=Yp[i]
        meano=sum_o/L
        meanp=sum_p/L
    
        numerator=0
        denumerator=0
        for i in range(L):
            numerator=numerator+abs(Yp[i]-Yo[i])
            denumerator=denumerator+abs(Yo[i]-meano)
        return 1-numerator/denumerator

    def rmean_squared_error(self, Yo, Yp, L):
        Sum=0
        for i in range(L):
            Sum=Sum+(Yo[i]-Yp[i])**2
        return math.sqrt(Sum/L)
    def writeOutPut(self,file):
        self.DataFrame=pd.DataFrame({'Time':self.cTime,'Forecast':self.cYp_test,'Observation':self.cY_test})
        self.MetricFrame=pd.DataFrame({'Section':[self.cTime[self.testSize*i] for i in range(self.section)], 'r2_score':self.metricsGrowth['r2_score'],'MAPE':self.metricsGrowth['MAPE'], 'SMAPE':self.metricsGrowth['SMAPE'],'RMSE':self.metricsGrowth['RMSE'],'CalcTime':self.compTime})        
        self.MetricFrame.to_csv("MetricFrame_"+file+".csv")
        self.DataFrame.to_csv("Forecast_"+file+".csv")
        
        
        
############################  Class Uncertainty for list of predictors ####################        
