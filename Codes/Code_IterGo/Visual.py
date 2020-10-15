#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:11:35 2019

@author: may706
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#s=[30] ## Sampling Window 
#p1=5
#t=[240]
#p2=240
#listSampleSize=[5, 10, 20, 30, 60, 120] 
#listTestSize=[300, 240, 120]#, 60, 30]     
listSampleSize=[10]#5, 10, 20, 30]#, 60, 120] 
listTestSize=[120]#300, 240, 120]#, 60, 30]     

Name='Adam_IterativeRNN_filter_5Real_s'
TitleMetric='/home/may706/LinuxBox/MetaData/MetricFrame_'+Name
TitleData='/home/may706/LinuxBox/MetaData/Uncertainty-Forecast_'+Name

FilesMetric=[[TitleMetric+str(sampleSize)+'_t'+str(testSize)+'.csv' for sampleSize in listSampleSize] for testSize in listTestSize]# for param in HyperParam]
FilesData=[[TitleData+str(sampleSize)+'_t'+str(testSize)+'.csv' for sampleSize in listSampleSize] for testSize in listTestSize]# for param in HyperParam]

MetricFrames=[[pd.read_csv(FilesMetric[i][j]) for j in range(len(listSampleSize))] for i in range(len(listTestSize))]# in FilesMetric]
DataFrames=[[pd.read_csv(FilesData[i][j]) for j in range(len(listSampleSize))] for i in range(len(listTestSize))]# in FilesMetric]


for i in range(len(listTestSize)):
    for j in range(len(listSampleSize)):
        plt.clf()
        fig, ax = plt.subplots(figsize=(14,8),frameon=True)
        plt.xlabel("$Time$ (seconds)")
        plt.ylabel("$P$ ")
        D1=DataFrames[i][j]       
        #plt.ylim(7000,8000)
        Tmax=D1.Time[-1]+240        
        plt.xlim(0,Tmax)
        plt.title("Numerous Predictors, "+Name+str(listSampleSize[j])+"_t"+str(listTestSize[i])+ ", Stage 15")
        print(FilesData[i][j])
#        print(i,j)
        for k in range(4):
        #    fTitle="/home/may706/LinuxBox/MetaData/Forecast_"+str(i+1)
            fTitle="Forecast_"+str(k+1)
            plt.plot(D1.Time,D1[fTitle])#"/home/may706/LinuxBox/MetaData/Forecast_"+str(i))
        
        x=[z for z in range(0,Tmax,480)]
        for z in range(0,Tmax,120):
            plt.axvspan(z,z+120,alpha=0.1*math.sin(0.0001*z))
        
        #for i in range(0,8000,480):
        plt.xticks(x)
        plt.grid(True)
        plt.savefig("/home/may706/LinuxBox/Images/"+Name+str(listSampleSize[j])+"_t"+str(listTestSize[i])+".png")
        
        plt.show()

####### Show Best Results ####### 
        
Name='Adam_IterativeRNN_filter_5Real_s'
TitleMetric='/home/may706/LinuxBox/MetaData/MetricFrame_'+Name
TitleData='/home/may706/LinuxBox/MetaData/Uncertainty-Forecast_'+Name

FilesMetric=[[TitleMetric+str(sampleSize)+'_t'+str(testSize)+'.csv' for sampleSize in listSampleSize] for testSize in listTestSize]# for param in HyperParam]
FilesData=[[TitleData+str(sampleSize)+'_t'+str(testSize)+'.csv' for sampleSize in listSampleSize] for testSize in listTestSize]# for param in HyperParam]

MetricFrames=[[pd.read_csv(FilesMetric[i][j]) for j in range(len(listSampleSize))] for i in range(len(listTestSize))]# in FilesMetric]
DataFrames=[[pd.read_csv(FilesData[i][j]) for j in range(len(listSampleSize))] for i in range(len(listTestSize))]# in FilesMetric]


for i in range(len(listTestSize)):
    for j in range(len(listSampleSize)):
        plt.clf()
        fig, ax = plt.subplots(figsize=(14,8),frameon=True)
        plt.xlabel("$Time$ (seconds)")
        plt.ylabel("$P$ ")
        D1=DataFrames[i][j]       
        #plt.ylim(7000,8000)
        Tmax=D1.Time[-1]+240        
        plt.xlim(0,Tmax)
        plt.title("Numerous Predictors, "+Name+str(listSampleSize[j])+"_t"+str(listTestSize[i])+ ", Stage 15")
        print(FilesData[i][j])
#        print(i,j)
        for k in range(4):
        #    fTitle="/home/may706/LinuxBox/MetaData/Forecast_"+str(i+1)
            fTitle="Forecast_"+str(k+1)
            plt.plot(D1.Time,D1[fTitle])#"/home/may706/LinuxBox/MetaData/Forecast_"+str(i))
        
        x=[z for z in range(0,Tmax,480)]
        for z in range(0,Tmax,120):
            plt.axvspan(z,z+120,alpha=0.1*math.sin(0.0001*z))
        
        #for i in range(0,8000,480):
        plt.xticks(x)
        plt.grid(True)
        plt.savefig("/home/may706/LinuxBox/Images/"+Name+str(listSampleSize[j])+"_t"+str(listTestSize[i])+".png")
        
        plt.show()
        
        
'''
plt.clf()
for i in range(5):
    plt.plot(listUncertainty[-1].cTime,listUncertainty[-1].cY) 
    plt.plot(listUncertainty[-1].cTime,listUncertainty[-1].Preds[0].cYp_test)
plt.show()
'''