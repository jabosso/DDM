# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:13:45 2019

@author: user
"""
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, BatchNormalization, Add, average
from BNS import BNS
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from scipy.misc import imsave
from keras.preprocessing import image
import os, os.path, math, time


def recurrent_block(i,y):
    i -=1
    x = MaxPooling2D((2,2), padding='same')(y)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    r = x 
    r = BatchNormalization()(r)
    if i < 0:
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)          
    else:    
        x= recurrent_block(i,x)     
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x) 
    x= BatchNormalization()(x)
    x= Add()([x,r])
    return x    
        
def build_rae(input_size):
    input_img = Input(shape=(input_size,input_size,1), name= 'image_input')    
    r = Conv2D(32,(3,3), activation='relu', padding='same', name='Conv1')(input_img)
   
    
    y = recurrent_block(2,r)
    
    x = UpSampling2D((2,2), name='upsample4')(y)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='Conv9')(x)
    
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')
    return autoencoder