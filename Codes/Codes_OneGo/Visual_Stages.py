#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:56:32 2019

@author: may706
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

testSize=120    ## testSize
sampleSize=10  ## HyperParam sampleSize 
ID=4230133468 ## ID of wells 
stages=[i for i in range(2,3)]
#stages=[15]
nStages=len(stages)
nPreds=5


#Name='IterativeMLP_test_ID'+str(ID)+'_stage' ## Name of Files 
#Name1='IterativeMLP_ID'+str(ID)+'_stage' ## Name of Files 
Name='OneGoMLP_test_ID'+str(ID)+'_stage' ## Name of Files 
Name1='OneGoMLP_ID'+str(ID)+'_stage' ## Name of Files 

Extension='_5Real_s'                ## 
TitleMetric='/home/may706/LinuxBox/MetaData/MetricFrame_'+Name
TitleData='/home/may706/LinuxBox/MetaData/Uncertainty-Forecast_'+Name
FilesMetric=[TitleMetric+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+'.csv' for stage in stages]# for param in HyperParam]
FilesData=[TitleData+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+'.csv' for stage in stages]    # for param in HyperParam]
PicPathForecast='/home/may706/LinuxBox/NewImages/Forecast_'+Name1
PicPathMetric='/home/may706/LinuxBox/NewImages/Metric_'+Name1

MetricFrames=[pd.read_csv(FilesMetric[i]) for i in range(len(stages))]# in FilesMetric]
DataFrames=[pd.read_csv(FilesData[i]) for i in range(len(stages))]# in FilesMetric]

                                                    ### Time series Plots ##
                                                    ### 1. Various Predictors
                                                    ### 2. Best Predictor 
                                                    ### 3. Worst Predictor 
                                                    ### 4. Uncertainty 
for i in range(nStages): 
    D1=DataFrames[i]       
    time=D1.Time.tolist()
    Tmax=int(time[-1]+240)
    print(Tmax)        

    ### Various Predictors ## 
    plt.clf()
    fig, ax = plt.subplots(figsize=(14,8),frameon=True)
    plt.xlabel("$Time$ (seconds)")
    plt.ylabel("$P$ ")
    #plt.ylim(7000,8000)
    plt.xlim(0,Tmax)
    stage=stages[i]
    plt.title("Various Predictors, Stage="+str(stage))
    for k in range(nPreds):
        fTitle="Forecast_"+str(k+1)
        plt.plot(D1.Time,D1[fTitle])#"/home/may706/LinuxBox/MetaData/Forecast_"+str(i))
    
    x=[z for z in range(0,Tmax,480)]
    for z in range(0,Tmax,120):
        plt.axvspan(z,z+120,alpha=0.1*math.sin(0.0001*z))
    
    #for i in range(0,8000,480):
    plt.xticks(x)
    plt.grid(True)#
    plt.savefig(PicPathForecast+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_VarPreds"+".png")
    
    plt.show()
            


    ### Best Predictor ## 
    plt.clf()
    fig, ax = plt.subplots(figsize=(14,8),frameon=True)
    ax.set_facecolor((1, 1, 1))
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.plot(D1.Time, D1.Observation,'black',linewidth=2,label='Observed Pressure')
    ax.plot(D1.Time, D1.BestForecast,'r',linewidth=1,label='Predicted Pressure')
    ax.legend()
    #c=mt_Frame[["Time"]]
#    plt.fill_between(D1.Time,D1.Forecast-2*D1.ConfidenceLevel,D1.Forecast+2*D1.ConfidenceLevel, alpha=0.5, color = '0.5')
    plt.xlabel("$Time$ (seconds)")
    plt.ylabel("$Pressure$")
#    plt.ylim(7000,8000)
    plt.xlim(0,Tmax)
    plt.title("Best Prediction, Stage="+str(stage))
    
    x=[i for i in range(0,Tmax,480)]
    for i in range(0,Tmax,120):
        plt.axvspan(i,i+120,alpha=0.1*math.sin(0.0001*i))
    plt.xticks(x)
    
    #for i in range(0,8000,480):
    plt.xticks(x)
    plt.grid(True)
    plt.savefig(PicPathForecast+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_BestForecast"+".png")
    
    plt.show()




    ### Worst Predictor ## 
    plt.clf()
    fig, ax = plt.subplots(figsize=(14,8),frameon=True)
    ax.set_facecolor((1, 1, 1))
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('k')
    #plt.show()
    #uncertain2=new_uncertain
    #ax.spines(color='black')
    #, label="Monthly average of Daily prediction")
    ax.plot(D1.Time, D1.Observation,'black',linewidth=2,label='Observed Pressure')
    ax.plot(D1.Time, D1.WorstForecast,'r',linewidth=1,label='Predicted Pressure')
    ax.legend()
    #c=mt_Frame[["Time"]]
#    plt.fill_between(D1.Time,D1.Forecast-2*D1.ConfidenceLevel,D1.Forecast+2*D1.ConfidenceLevel, alpha=0.5, color = '0.5')
    plt.xlabel("$Time$ (seconds)")
    plt.ylabel("$Pressure$")
#    plt.ylim(7000,8000)
    plt.xlim(0,Tmax)
    plt.title("Worst Prediction, Stage="+str(stage))
    
    x=[i for i in range(0,Tmax,480)]
    for i in range(0,Tmax,120):
        plt.axvspan(i,i+120,alpha=0.1*math.sin(0.0001*i))
    plt.xticks(x)
    
    #for i in range(0,8000,480):
    plt.xticks(x)
    plt.grid(True)
    plt.savefig(PicPathForecast+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_WorstForecast"+".png")
    
    plt.show()

    

    ### Uncertainty Window ## 
    plt.clf()
    fig, ax = plt.subplots(figsize=(14,8),frameon=True)
    ax.set_facecolor((1, 1, 1))
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('k')
    #plt.show()
    #uncertain2=new_uncertain
    #ax.spines(color='black')
    #, label="Monthly average of Daily prediction")
    ax.plot(D1.Time, D1.Observation,'black',linewidth=2,label='Observed Pressure')
    ax.plot(D1.Time, D1.Forecast,'r',linewidth=1,label='Predicted Pressure')
    ax.legend()
    plt.fill_between(D1.Time,D1.Forecast-2*D1.ConfidenceLevel,D1.Forecast+2*D1.ConfidenceLevel, alpha=0.5, color = '0.5')
    plt.xlabel("$Time$ (seconds)")
    plt.ylabel("$Pressure$")
#    plt.ylim(7000,8000)
    plt.xlim(0,Tmax)
    plt.title("Uncertainty, Stage="+str(stage))
    
    x=[i for i in range(0,Tmax,480)]
    for i in range(0,Tmax,120):
        plt.axvspan(i,i+120,alpha=0.1*math.sin(0.0001*i))
    plt.xticks(x)
    
    #for i in range(0,8000,480):
    plt.xticks(x)
    plt.grid(True)
    plt.savefig(PicPathForecast+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_Uncertainty-AvForecast"+".png")
    
    plt.show()



###########################################################################################
#### Performance Metrics ##################################################################    
                                    ##### Best Forecast ########
for i in range(nStages): 
## Plotting the metrics performance ## 
    
    plt.clf()
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(14,8),frameon=True)
    axs[0].set_title("RMSE, stage = "+str(stages[i]))
    #axs[0,0].set_ylim(-1,1)
    metric=MetricFrames[i]
    axs[0].plot(metric.Section,metric.Best_RMSE)
    axs[0].legend()
    
        
    axs[1].set_title("NRMSE, stage = "+str(stages[i]))    
    metric=MetricFrames[i]
#    title='Window='+str(HyperParam[i])
    axs[1].plot(metric.Section,metric.Best_NRMSE)#, label=title)
    axs[1].legend()
        
    #axs[0,0].set_ylim([-1,1])
    
    plt.savefig(PicPathMetric+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_BestForecast"+".png")
    plt.show()    


                                    ### Average Forecast ## 

for i in range(nStages): 
## Plotting the metrics performance ## 
    
    plt.clf()
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(14,8),frameon=True)
    axs[0].set_title("RMSE, stage = "+str(stages[i]))
    #axs[0,0].set_ylim(-1,1)
    metric=MetricFrames[i]
    axs[0].plot(metric.Section,metric.RMSE)
    axs[0].legend()
    
    axs[1].set_title("NRMSE, stage = "+str(stages[i]))    
    metric=MetricFrames[i]
#    title='Window='+str(HyperParam[i])
    axs[1].plot(metric.Section,metric.NRMSE)#, label=title)
    axs[1].legend()
                
#    axs[0].set_ylim([-1,1])
    
    plt.savefig(PicPathMetric+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_avForecast"+".png")
    plt.show()    
