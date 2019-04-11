# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:06:31 2019

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


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray =(0.2989*r + 0.5870*g + 0.1140*b)
    return gray

class BNS():
    def __init__(self,input_size):
        self.input_size =input_size
    
    
    def getPatch(self, path,kernel_size):
        img = image.load_img(path, target_size=(1320,1030))#
        np_img = image.img_to_array(img)
        gray_img = rgb2gray(np_img)
        group_of_patch= np.empty(kernel_size)    
        patch = np.arange(kernel_size[1]*kernel_size[2]).reshape(kernel_size)
        nx=880//self.input_size
        ny=1120//self.input_size
        for j in range(ny):
            y=self.input_size*j
            for i in range(nx):
                x = self.input_size*i        
                patch[0] = gray_img[y:y+kernel_size[1], x:x+kernel_size[2]]           
                group_of_patch = np.concatenate((group_of_patch, patch),axis=0)            
        return group_of_patch[1:]

    def get_patch_from_directory(self, path, kernel_size):
        data_list = os.listdir(path)
        X_set = np.empty(kernel_size)    
        for i in range(len(data_list)):
            img_path = os.path.join(path, data_list[i])
            X_set = np.concatenate((X_set, self.getPatch(img_path, kernel_size)),axis=0)
        return X_set[1:]
    def load_data(self):
        path = os.path.join('train')  
        train_set = np.empty((1,self.input_size,self.input_size))
        train_set =self.get_patch_from_directory(path,(1,self.input_size,self.input_size))
        path = os.path.join('train_cleaned')  
        train_label = np.empty((1,self.input_size,self.input_size))
        train_label =self.get_patch_from_directory(path,(1,self.input_size,self.input_size))
        train_set = train_set.astype('float32') / 255.0
        train_label =train_label.astype('float32') / 255.0    
        train_set = np.reshape(train_set, (len(train_set), self.input_size, self.input_size, 1)) 
        train_label = np.reshape(train_label, (len(train_label), self.input_size, self.input_size, 1))  
        im_test = np.reshape(train_set[20],(self.input_size,self.input_size))
    
        path = os.path.join('validation')  
        validation_set = np.empty((1,self.input_size,self.input_size))
        validation_set =self.get_patch_from_directory(path,(1,self.input_size,self.input_size))
        path = os.path.join('validation_cleaned')  
        validation_label = np.empty((1,self.input_size,self.input_size))
        validation_label =self.get_patch_from_directory(path,(1,self.input_size,self.input_size))
        validation_set = validation_set.astype('float32') / 255.0
        validation_label =validation_label.astype('float32') / 255.0    
        validation_set = np.reshape(validation_set, (len(validation_set), self.input_size, self.input_size, 1)) 
        validation_label = np.reshape(validation_label, (len(validation_label), self.input_size, self.input_size, 1))  
        
        train_set= np.nan_to_num(train_set)
        #train_label= np.nan_to_num(train_label)
    
        validation_set= np.nan_to_num(validation_set)
        #validation_label= np.nan_to_num(validation_label)
        
        return train_set, train_label, validation_set, validation_label
    