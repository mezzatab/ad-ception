#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import Preprocess 

testSize=120    ## testSize
sampleSize=2#ls Metric0  ## HyperParam sampleSize 
stages=[i for i in range(2,21)]
#stages.append(15)
#stages=[15]0
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
PicPath='/home/may706/LinuxBox/BasicImages/'
MetricFrames=[pd.read_csv(FilesMetric[i]) for i in range(len(stages))]# in FilesMetric]
DataFrames=[pd.read_csv(FilesData[i]) for i in range(len(stages))]# in FilesMetric]


########### Raw data ####


df=Preprocess.readFiles(str(ID))
#print(df)
df=Preprocess.Preprocess(df,list_Params)#,15)


###################################

                                                    ### Time series Plots ##
                                                    ### 1. Various Predictors
                                                    ### 2. Best Predictor 
                                                    ### 3. Worst Predictor 
                                                    ### 4. Uncertainty 
for i in range(nStages):
    
    
    
    
    ########################### Observatns plots ###################################################]
    scaledFrame=Preprocess.diff_Scale_midfil(df,stages[i])
    nframe=scaledFrame[['Time', 'bPropConc', 'frConc','sRate','P', 'prConc']].values


    
    plt.clf()
    fig, axs = plt.subplots(nrows=5,ncols=1,figsize=(14,8),frameon=True)
    #plt.xlabel("Section")
    #plt.ylabel("$Pressure$ ")
    #plt.ylim(7000,8000)
    #plt.xlim(0,8000)
    #plt.title("With and Without P, CNN-RNN, Stage 15")
    #axs[0].set_title("Wellhead Pressure")
#    axs[0].plot(nframe[:,0],nframe[:,5], label="Differebce Pressure",color='r')
    #axs[0,0].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.r2_score, label="Reg_NG")
#    axs[0].legend()
#    axs[0].set_ylim(-10,10)
    
#    x=[tmax for tmax in range(0,Tmax,dtmax)]
#    for tmax in range(0,Tmax,dt):
#        plt.axvspan(tmax,tmax+dt,alpha=0.1*math.sin(0.0001*tmax))
#    plt.xticks(x)

    
    dtmax=600
    dt=240
    
    stage=stages[i]    
    D1=DataFrames[i]       
    time=D1.Time.tolist()
    Tmax=int(time[-1]+240)
     
    
    axs[0].plot(nframe[:,0],nframe[:,4], label="Wellhead Pressure",color='r')
    #axs[0,0].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.r2_score, label="Reg_NG")
    axs[0].legend()
     
    #axs[0].set_ylim(-10,10)
    
    #axs[0].plot(nframe[:,0],nframe[:,4], label="Wellhead Pressure",color='r')
    #axs[0,0].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.r2_score, label="Reg_NG")
    #axs[0].legend()
    #axs[0,0].set_ylim([-1,1])
    
    #axs[1].set_title("bottom hole proppant concentration")
    axs[1].plot(nframe[:,0],nframe[:,1], label="bottom hole proppant concentration")
    #axs[1,0].plot(MetricFrame_Reg_nd.Section,MetricFrame_Reg_nd.MAPE/100, label="Reg")
    #axs[0,1].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.MAPE, label="Reg_NG")
    axs[1].legend()
    
     
    
    #axs[2].set_title("proppant concentration")
    axs[2].plot(nframe[:,0],nframe[:,5], label="proppant concentration")
    #axs[2].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.SMAPE, label="Reg_NG")
    axs[2].legend()

     
    
    #axs[2].set_title("proppant concentration")
    axs[3].plot(nframe[:,0],nframe[:,2], label="frConc")
    #axs[2].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.SMAPE, label="Reg_NG")
    axs[3].legend()

     
    
    axs[4].plot(nframe[:,0],nframe[:,3], label="Slurry Rate")
    #axs[2].plot(MetricFrame_Reg_nd_NG.Section,MetricFrame_Reg_nd_NG.SMAPE, label="Reg_NG")
    axs[4].legend()
#    x=[tmax for tmax in range(0,Tmax,dtmax)]
#    for tmax in range(0,Tmax,dt):
#        plt.axvspan(tmax,tmax+dt,alpha=0.1*math.sin(0.0001*tmax))
#    plt.xticks(x)
    
    
    plt.savefig(PicPath+"MultiTimeSeries_Stage"+str(stages[i])+".png")
    plt.show()
    
    

    ### Uncertainty Window ##
#    plt.xlim(0,Tmax)
    stage=stages[i]    
    D1=DataFrames[i]       
    time=D1.Time.tolist()
    Tmax=int(time[-1]+240)
    minP=D1.Observation.min()
    maxP=D1.Observation.max()
    DeltaP=maxP-minP
    minP=minP-0.2*DeltaP
    maxP=maxP+0.2*DeltaP
#    print(Tmax)        
    
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
    plt.ylim(minP,maxP)
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




    
    
        ##############################################################################
#    D1=DataFrames[i]       
#    time=D1.Time.tolist()
#    Tmax=int(time[-1]+240)
#    print(Tmax)        

    ### Various Predictors ## 
    plt.clf()
    fig, ax = plt.subplots(figsize=(14,8),frameon=True)
    plt.xlabel("$Time$ (seconds)")
    plt.ylabel("$Pressure$ ")
    #plt.ylim(7000,8000)
    plt.title("Various Predictors, Stage="+str(stage))
    for k in range(nPreds):
        fTitle="Forecast_"+str(k+1)
        plt.plot(D1.Time,D1[fTitle])#"/home/may706/LinuxBox/MetaData/Forecast_"+str(i))
    
    x=[z for z in range(0,Tmax,480)]
    for z in range(0,Tmax,120):
        plt.axvspan(z,z+120,alpha=0.1*math.sin(0.0001*z))
    
    #for i in range(0,8000,480):
    plt.xticks(x)
    plt.xlim(0,Tmax)
    plt.grid(True)#
    plt.ylim(minP,maxP)
    
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
    plt.ylim(minP,maxP)
    
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
#    plt.xticks(x)
    plt.grid(True)
    plt.ylim(minP,maxP)
    
    plt.savefig(PicPathForecast+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_WorstForecast"+".png")
    
    plt.show()

    

print(nStages)
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
    plt.ylim(-1,1)
    
        
    #axs[0,0].set_ylim([-1,1])
#    plt.ylim(minP,maxP) 
#    print(PicPathMetric+str(stages[i])+Extension+str(sampleSize)+'_t'+str(testSize)+"_BestForecast"+".png")
    plt.savefig(PicPathMetric+str(stages[i])+Extension+str(sampleSize)+'_t'+str(testSize)+"_BestForecast"+".png")
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
    plt.ylim(-1,1)
                
#    axs[0].set_ylim([-1,1])
#    print(PicPathMetric+str(stage)+Extension+str(sampleSize)+'_t'+str(testSize)+"_avForecast"+".png")
    plt.savefig(PicPathMetric+str(stages[i])+Extension+str(sampleSize)+'_t'+str(testSize)+"_avForecast"+".png")
    plt.show()    
