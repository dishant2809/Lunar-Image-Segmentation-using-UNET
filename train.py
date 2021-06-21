from data_loader import Dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Dropout, BatchNormalization,Concatenate,ReLU,LeakyReLU,Activation,Input,MaxPool2D
from keras.layers import Add
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from main import UNET
from data_loader import Dataset
from utils import *
from tqdm import *

im = 'C:/Users/vedant/ML&AI/Project/vedant/Semantic segmentation dataset/Tile 1/images'
mk = 'C:/Users/vedant/ML&AI/Project/vedant/Semantic segmentation dataset/Tile 1/masks'



# Defining custom model fit in simple way possible

def train(data, model, optimizer, loss, metrics):
    loop = tqdm(data,leave=True)
    for idx,(img,mask) in enumerate(loop):
        with tf.GradientTape() as tape:
            ypred = model(img)
            ls = loss(mask,ypred)
            #metrics.reset_states()
        gradients = tape.gradient(ls, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        
        metrics.update_state(mask,ypred)
        print({m.name:m.result() for m in metrics})
        
        metrics.reset_states()
        return i
        #return gradients


epochs = 50


# Direct using tf.Gradienttape to train model

for i in range(epochs):
    loop = tqdm(Dataset(im,mk),leave=True)
    for idx, (img,mask) in enumerate(loop):
        with tf.GradientTape() as tape:
            ypred = model(img)
            ls = loss(mask,ypred)
            gradients = tape.gradient(ls, model.trainable_variables)
        
        #metrics.reset_states()
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #metrics.update_state(mask,ypred)
        
        print(i)
        
        
            







model = UNET()
optimizer =  tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.CategoricalAccuracy()

train(Dataset(im, mk),model,optimizer,loss,metrics)


