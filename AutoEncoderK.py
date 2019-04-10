# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:47:43 2019

@author: user
"""
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, BatchNormalization
from BNS import BNS
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from scipy.misc import imsave
from keras.preprocessing import image
import os, os.path, math, time

def build_autoencoder(input_size):
    input_img = Input(shape=(input_size,input_size,1), name= 'image_input')
    
    x = Conv2D(32,(3,3), activation='relu', padding='same', name='Conv1')(input_img)
 
    x = MaxPooling2D((2,2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2')(x)

    x = MaxPooling2D((2,2), padding='same', name='pool2')(x) 
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3')(x)

    x = MaxPooling2D((2,2), padding='same', name='pool3')(x) 
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv4')(x)
    
    x = MaxPooling2D((2,2), padding='same', name='pool4')(x) 
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv5')(x)
  
    x = UpSampling2D((2,2), name='upsample1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv6')(x)

    x = UpSampling2D((2,2), name='upsample2')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv7')(x)

    x = UpSampling2D((2,2), name='upsample3')(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv8')(x)

    x = UpSampling2D((2,2), name='upsample4')(x)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='Conv9')(x)
    
    
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')
    return autoencoder


    
    


input_size = 512
autoencoder = build_autoencoder(input_size)
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
                epochs=300,
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