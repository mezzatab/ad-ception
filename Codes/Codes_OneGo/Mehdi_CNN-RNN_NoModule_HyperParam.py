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

np.random.seed(42)
set_random_seed(42)


listNames=["4230133468","4247536931","4247537091","4247537106","l4247537154","4247537186"]
list_Params=['date_time','stage_number','prop_conc_1','bh_prop_conc','fr_conc_2','slurry_rate','wellhead_pressure_1']


df=Preprocess.readFiles("4230133468")
#print(df)
df=Preprocess.Preprocess(df,list_Params)#,15)
scaledFrame=Preprocess.diff_Scale(df,15)
scaledFrame.P=scaledFrame.P-scaledFrame.P.iloc[0]
nframe=scaledFrame[['Time','prConc', 'frConc','sRate','P', 'bPropConc']].values

config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':32})#device_count = {'GPU':1, 'CPU':32})
#config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

## Metric frames ## 


#config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':32})
#tf.config.gpu.set_per_process_memory_fraction(0.4)
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)
#print(device_lib.list_local_devices())


class WellheadPressurePrediction:

    def __init__(self, df, sample_size, test_size, engine="default"):
        self.df = df
        (self.totalTime, xAxis)=df.shape
        self.nFeatures=4
        self.sampleSize=sample_size
        self.testSize=test_size
        self.cY_test=None
        self.cYp_test=None  
        self.cTime=None        
        self.compTime=[]
        self.preFry=np.zeros(self.testSize)
        
        self.Engine = {
            "default":setup.setup,
            "1layer":setup.setup1layer,            
            "2layer":setup.setup2layer,                              
            "2layerRNN":setup.setup2layerRNN                                           
        }
        
        self.model=self.Engine[engine](4, sample_size, test_size, 'mse')
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
            self.metricsGrowth[key].append(self.METRICS[key](self.Y_test[:,0],self.Yp_test[:,0],self.rT))
            
    def train(self,T):
        self.T=T
        self.preprocess()

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1, mode='min')
        self.history=self.model.fit(self.X[self.sampleSize:self.T-self.sampleSize], self.Y[self.sampleSize:self.T-self.sampleSize], epochs=100, batch_size=200, verbose=0, callbacks=[early_stop])
        
##########    Predict with no ground Truth #######################            

#### The assumption is T+sampleSize is existing in data file #### 

    def predict(self):
        self.section=self.section+1
        
        self.tX=self.scalerX.transform(self.df[self.T-self.sampleSize:self.T+self.sampleSize,:])        
        
        self.testX=np.zeros((1,self.sampleSize,4))        ## One point for prediction 
        
        
        self.testX[0,:,0]=self.tX[self.sampleSize:2*self.sampleSize,1]#.tolist()
        self.testX[0,:,1]=self.tX[self.sampleSize:2*self.sampleSize,2]#.tolist()            
        self.testX[0,:,2]=self.tX[self.sampleSize:2*self.sampleSize,3]#.tolist()     # #               
        self.testX[0,:,3]=self.tX[:self.sampleSize,4]#.tolist()     # # Pressure Items # #        

        
        ### If the prediction goes byond the existing data file ### rT != testSize 
        self.rT=self.testSize
        if self.T+self.testSize > self.totalTime:
            self.rT=self.totalTime-self.T
        
        self.timeTest=np.zeros((self.rT))        
        for i in range(self.rT):
            self.timeTest[i]=self.df[self.T+i,0]
#        self.Time=self.Time.tolist()
        
        
        
        #### Prediction Engine ####                 
        self.preFry=self.model.predict(self.testX)
        #self.preFry=self.preFry[:self.rT] ## 
        self.Yp_test=self.scalerY.inverse_transform(self.preFry.reshape(-1,1))
        self.Yp_test=self.Yp_test[:self.rT]
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
        
class Uncertainty:

    def __init__(self, df, sampleSize, testSize, nPredict, nSections=10):
        self.Preds=[]
        (self.tTime,xAxis)=df.shape
        self.nPredict=nPredict
        self.sampleSize=sampleSize
        self.testSize=testSize
        self.nSections=nSections
#        self.nSections=1+int((self.tTime-self.sampleSize)/(self.testSize))
        for i in range(nPredict):
            self.Preds.append(WellheadPressurePrediction(df, self.sampleSize, self.testSize, '2layerRNN'))
            
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
        
    def metric(self,X,Y):
        for key in self.METRICS.keys():
            self.metricsGrowth[key].append(self.METRICS[key](X,Y))#,self.testSize))
        
            
            
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
            
            
        for i in range(self.nSections-1):
            self.metric(self.cY[i*self.testSize:(i+1)*self.testSize],self.avgCalc[i*self.testSize:(i+1)*self.testSize])
        i=self.nSections-1
        self.metric(self.cY[i*self.testSize:i*self.testSize+self.Preds[0].rT],self.avgCalc[i*self.testSize:i*self.testSize+self.Preds[0].rT])            
            
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
        self.Dict={'Time':self.cTime,'Observation':self.cY,'Forecast':self.avgCalc,'ConfidenceLevel':2*self.SigmaCalc}
        self.ForecastDict={'Forecast_'+str(i+1):self.Preds[i].cYp_test for i in range(self.nPredict)}
        self.Dict.update(self.ForecastDict)
        self.DataFrame=pd.DataFrame(self.Dict)#{'Time':self.cTime,'Observation':self.cY,'Forecast':self.avgCalc,'ConfidenceLevel':2*self.SigmaCalc})#,ForecastDict})
        self.DataFrame.to_csv("/home/may706/LinuxBox/MetaData/Uncertainty-Forecast_"+file+".csv")
        self.MetricFrame=pd.DataFrame({'Section':[self.Preds[0].cTime[self.testSize*i] for i in range(self.nSections)], 'NMSE':self.metricsGrowth['NMSE'], 'NRMSE':self.metricsGrowth['NRMSE'],'MAPE':self.metricsGrowth['MAPE'], 'SMAPE':self.metricsGrowth['SMAPE'],'RMSE':self.metricsGrowth['RMSE'],'CalcTime':self.compTime})        
        self.MetricFrame.to_csv("/home/may706/LinuxBox/MetaData/MetricFrame_"+file+".csv")
        


        
                                        ####### Int main ####### 

## COnsider dP ## 
#testSize=120 
#listSampleSize=[10, 20, 30, 60, 120, 300] 
listSampleSize=[5, 10, 20, 30, 60, 120] #, 180, 240, 300]
listTestSize=[300, 240, 120, 60]#, 30]    #5, 10, 20, 30, 60, 120, 180, 240, 300] 
#listSampleSize=[60, 120, 180, 240, 300] 
#testSize=[120 for i in range(len(listSampleSize))]
nPredictant=15
listUncertainty=[]
for testSize in listTestSize:
    for sampleSize in listSampleSize:
        print(testSize,sampleSize)
        (tTime,z)=nframe.shape
        t0=sampleSize*3
        finalSection1=1+int((tTime-t0-sampleSize)/(testSize))
        finalSection2=1+int((tTime-t0+sampleSize-testSize)/(testSize))
        nSections=min(finalSection1,finalSection2)    
        uncertain=Uncertainty(nframe, sampleSize, testSize, nPredictant, nSections)
        listUncertainty.append(uncertain)
        listUncertainty[-1].ListRun()
        listUncertainty[-1].uncertaintyAnalysis()
        listUncertainty[-1].writeOutPut('Adam_Default_2layerRNN_s'+str(sampleSize)+'_t'+str(testSize))
    


        
        
                
