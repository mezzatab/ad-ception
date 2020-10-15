## Start from The beginning ## 
## Time series from (0,T) using sliding window turn into small images with output Pressure ## 
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
#from tensorflow.keras.utils import vis_utils 
np.random.seed(42)
set_random_seed(42)


def setup(nFeatures, sampleSize, testSize,  LOSS='mse'): ### 3layers CNN-RNN Models 
    #print(self.nFeatures)
    model = Sequential()
        
    model.add(Conv1D(10,nFeatures,input_shape=(sampleSize,nFeatures),padding='same', activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))
        
    model.add(Conv1D(10,nFeatures,padding='same', activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))
        
    model.add(Conv1D(10,nFeatures,padding='same', activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))
        
    model.add(SimpleRNN(10, return_sequences = True))
    model.add(SimpleRNN(10))

    model.add(Dense(testSize))
    model.compile(loss=LOSS, optimizer='adam')
    return model 


def setupMLP(nFeatures, sampleSize, testSize,  LOSS='mse'):
    model = Sequential()
    size=10        
    model.add(Dense(size, input_dim=nFeatures*sampleSize, activation = 'relu'))
    model.add(Dense(testSize))
    model.compile(loss=LOSS, optimizer='adam')
    return model



def setup1layer(nFeatures, sampleSize, testSize,  LOSS='mse'): ## One Layer CNN-RNN Model 
    
    model = Sequential()
    CNN_Screen=1
    RNN_size=10
    model.add(Conv1D(1,CNN_Screen,input_shape=(sampleSize, nFeatures),padding='same', activation = 'relu'))
    model.add(SimpleRNN(RNN_size))
    model.add(Dense(testSize))
#    opt=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#    opt=keras.optimizers.Nadam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)    
    model.compile(loss=LOSS, optimizer='adam')
    return model 

def setup2layerRNN(nFeatures, sampleSize, testSize,  LOSS='mse'): ## One Layer CNN-RNN Model 
    
    model = Sequential()
    CNN_Screen=1
    RNN_size=10
#    model.add(Conv1D(1,CNN_Screen,input_shape=(sampleSize, nFeatures),padding='same', activation = 'relu'))
    model.add(SimpleRNN(RNN_size, input_shape=(sampleSize, nFeatures), return_sequences = True))    
    model.add(SimpleRNN(RNN_size))
    model.add(Dense(testSize))
#    opt=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#    opt=keras.optimizers.Nadam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)    
    model.compile(loss=LOSS, optimizer='adam')
    return model 


def setup2layer(nFeatures, sampleSize, testSize,  LOSS='mse'): ## One Layer CNN-RNN Model with RNN size = testSize 
    
    model = Sequential()
    CNN_Screen=1
    RNN_size=10
    model.add(Conv1D(1,CNN_Screen,input_shape=(sampleSize, nFeatures),padding='same', activation = 'relu'))
    model.add(SimpleRNN(testSize))
#    model.add(Dense(testSize))
    model.compile(loss=LOSS, optimizer='adam')
    return model 

def setupRNN(nFeatures, sampleSize, testSize,  LOSS='mse'): ## One Layer CNN-RNN Model with RNN size = testSize 
    
    model = Sequential()
    RNN_size=10
    model.add(SimpleRNN(RNN_size, input_shape=(sampleSize, nFeatures), return_sequences = True))        
    model.add(SimpleRNN(RNN_size, input_shape=(sampleSize, nFeatures)))#, input_shape=(sampleSize, nFeatures)))
    model.add(Dense(1))
    model.compile(loss=LOSS, optimizer='adam')
    return model 

