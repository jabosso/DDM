# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:46:00 2019

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
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder
input_size = 512
autoencoder = build_rae(input_size)
autoencoder.summary()


checkpointer = ModelCheckpoint(filepath= os.path.join('checkpoints','{epoch:03d}-{val_loss:.3f}.hdf5'),
                               verbose=1,
                               save_best_only=True)
tb = TensorBoard(log_dir=os.path.join('logs'))
early_stopper = EarlyStopping(patience=11)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('logs','training-' + \
                                    str(timestamp) + '.log'))

bNS = BNS(input_size)   
x_train, y_train, x_test, y_test = bNS.load_data()







autoencoder.fit(x_train, y_train,  batch_size=8, validation_data=(x_test, y_test),
                epochs=1,
                 callbacks=[tb,early_stopper,csv_logger,checkpointer])



path_test = os.path.join('test/13.png') 
test_im = np.empty((1,input_size,input_size))
test_im = bNS.getPatch(path_test,(1, input_size, input_size))

print(test_im.shape)
patch = test_im[1]
plt.imshow(patch)   
patch =np.reshape(patch,(1,input_size,input_size,1))
clean_patch = autoencoder.predict(patch)

clean_patch = np.reshape(clean_patch, (input_size, input_size))    
print(clean_patch.shape)  

imsave(str(input_size)+'.png',clean_patch)  