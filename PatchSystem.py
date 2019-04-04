# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:15:18 2019

@author: user
"""
from keras import backend as K
from keras.models import Model
from scipy.misc import imsave
from keras.preprocessing import image
import os, os.path, math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.callbacks import TensorBoard


class AutoEncoder():
    def __init__(self,input_size):
        input_img = Input(shape=(input_size,input_size,1)) 
        
        x = Conv2D(32,(6,6), activation='relu', padding= 'same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(32,(6,6), activation= 'relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(6,(3,3), activation= 'relu', padding='same')(x)
        encoded = MaxPooling2D((2,2), padding='same')(x)
        
        x = Conv2D(32, (6, 6), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (6, 6), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(28, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (6,6), activation='sigmoid', padding='same')(x)
        self.unit = Model(input_img, decoded)
        self.unit.compile(optimizer='adam', loss='binary_crossentropy')
        print(self.unit.summary())

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray =(0.2989*r + 0.5870*g + 0.1140*b)
    return gray 

def getPatch(path,kernel_size,input_size):    
    img = image.load_img(path, target_size=(1120,880))
    np_img = image.img_to_array(img)
    gray_img = rgb2gray(np_img)
    group_of_patch= np.empty(kernel_size)    
    patch = np.arange(kernel_size[1]*kernel_size[2]).reshape(kernel_size)
    nx=880//input_size
    ny=1120//input_size
    for j in range(ny):
        y=input_size*j
        for i in range(nx):            
            x = input_size*i        
            patch[0] = gray_img[y:y+kernel_size[1], x:x+kernel_size[2]]           
            group_of_patch = np.concatenate((group_of_patch, patch),axis=0)            
    return group_of_patch[1:]

def get_patch_from_directory(path, kernel_size,input_size):
    data_list = os.listdir(path)
    X_set = np.empty(kernel_size)    
    for i in range(len(data_list)):
        img_path = os.path.join(path, data_list[i])
        X_set = np.concatenate((X_set, getPatch(img_path, kernel_size,input_size)),axis=0)
    return X_set[1:]

    
def main(input_size):   
    path = os.path.join('ImgSporche/Train_set')  
    train_set = np.empty((1,input_size,input_size))
    train_set =get_patch_from_directory(path,(1,input_size,input_size),input_size)
    path = os.path.join('ImgPulite/Train_Label')  
    train_label = np.empty((1,input_size,input_size))
    train_label =get_patch_from_directory(path,(1,input_size,input_size),input_size)
    train_set = train_set.astype('float32') / 255.0
    train_label =train_label.astype('float32') / 255.0    
    train_set = np.reshape(train_set, (len(train_set), input_size, input_size, 1)) 
    train_label = np.reshape(train_label, (len(train_label), input_size, input_size, 1))  
    im_test = np.reshape(train_set[20],(input_size,input_size))
    
    path = os.path.join('ImgSporche/Validation_set')  
    validation_set = np.empty((1,input_size,input_size))
    validation_set =get_patch_from_directory(path,(1,input_size,input_size),input_size)
    path = os.path.join('ImgPulite/Validation_Label')  
    validation_label = np.empty((1,input_size,input_size))
    validation_label =get_patch_from_directory(path,(1,input_size,input_size),input_size)
    validation_set = validation_set.astype('float32') / 255.0
    validation_label =validation_label.astype('float32') / 255.0    
    validation_set = np.reshape(validation_set, (len(validation_set), input_size, input_size, 1)) 
    validation_label = np.reshape(validation_label, (len(validation_label), input_size, input_size, 1))  
    
    
    
    
    autoencoder = AutoEncoder(input_size)
    train_set= np.nan_to_num(train_set)
    train_label= np.nan_to_num(train_label)
    
    validation_set= np.nan_to_num(validation_set)
    validation_label= np.nan_to_num(validation_label)
    
    autoencoder.unit.fit(train_set, train_label,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(validation_set, validation_label),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    path_test = os.path.join('ImgSporche/Test_set/19.jpeg') 
    test_im = np.empty((1,input_size,input_size))
    test_im = getPatch(path_test,(1, input_size, input_size),input_size)
    group_of_patch= np.empty((1,input_size,input_size,1))
    for patch in test_im :
        patch =np.reshape(patch,(1,input_size,input_size,1))
        clean_patch = autoencoder.unit.predict(patch)
        group_of_patch = np.concatenate((group_of_patch, clean_patch),axis=0)
    print(group_of_patch.shape)  
    group_of_patch = group_of_patch[1:]
    
    a = (1120//input_size)*input_size
    b = (880//input_size)*input_size
    cleaned = np.reshape(group_of_patch,(1,a,b))
    imsave(str(input_size)+'.png',cleaned[0])  
if __name__ == '__main__' :
    for i in range(10):
        x= 28+8*i
        main(x)    