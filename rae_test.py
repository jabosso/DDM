# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:01:19 2019

@author: user
"""
import os , os.path, h5py
from RAE_model import build_rae

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, BatchNormalization, Add, average
from BNS import BNS
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from scipy.misc import imsave
from keras.preprocessing import image
import os, os.path, math, time

input_size= 256

bNS = BNS(input_size)

rae = build_rae(input_size)
#rae.load_weights('checkpoints/005-0.012.hdf5')

path_test = os.path.join('01.jpeg') 
test_im = np.empty((1,input_size,input_size))
test_im = bNS.getPatch(path_test,(1, input_size, input_size))

print(test_im.shape)
patch = test_im[1]
plt.imshow(patch)   
patch =np.reshape(patch,(1,input_size,input_size,1))
checkp = os.listdir('checkpoints')
i=0
for checkpo in checkp :
    i= i+1
    print(checkpo)
    rae.load_weights('checkpoints/'+checkpo)
    clean_patch = rae.predict(patch)
    clean_patch = np.reshape(clean_patch, (input_size, input_size))    
    print(clean_patch.shape)  
    imsave(str(i)+'-'+str(input_size)+'.png',clean_patch)  