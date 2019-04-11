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

from RAE_model import build_rae

    

input_size = 256
autoencoder = build_rae(input_size)
autoencoder.summary()


checkpointer = ModelCheckpoint(filepath= os.path.join('checkpoints','{epoch:03d}-{val_loss:.3f}.hdf5'),
                               verbose=1,
                               save_best_only=True)
tb = TensorBoard(log_dir=os.path.join('logs'))
early_stopper = EarlyStopping(patience=3)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('logs','training-' + \
                                    str(timestamp) + '.log'))

bNS = BNS(input_size)   
x_train, y_train, x_test, y_test = bNS.load_data()







autoencoder.fit(x_train, y_train,  batch_size=8, validation_data=(x_test, y_test),
                epochs=20,
                 callbacks=[tb,early_stopper,csv_logger,checkpointer])



 