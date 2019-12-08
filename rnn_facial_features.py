# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:56:25 2019

@author: melbs
"""

# import libraries 
import pandas as pd
import numpy as np 
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Import data 
#filepath = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\output\\csv"
file1 = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\output\\csv\\31_0.csv"
file2 = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\output\\csv\\31_10.csv"
file3 = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\output\\csv\\34_0.csv"
file4 = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\output\\csv\\34_10.csv"

# training and validation sets 
df1 = pd.read_csv(file1,header=0,index_col=0) 
df2 = pd.read_csv(file2,header=0,index_col=0) 
df3 = pd.read_csv(file3,header=0,index_col=0) 
df4 = pd.read_csv(file4,header=0,index_col=0) 
# training and validation dataframe
df = pd.concat([df1,df2,df3,df4],axis=0)

# Define variables 
time = df.time
dt = time.diff(periods=1).fillna(0).values
df.replace(-1,np.NaN,inplace=True)
df.interpolate(inplace=True)
feat = df[df.columns[11:145]]
labels = df.mood.values

# Define X, y 
X = feat
y = labels

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.3, random_state=0)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0],1, val_X.shape[1]))
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)
 
# design network
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(10,
               input_shape=(train_X.shape[1], train_X.shape[2]))) # returns a sequence of vectors of dimension 32
#model.add(LSTM(1, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(2))  # return a single vector of dimension 32
model.add(Dense(1))#, activation='softmax'))

model.compile(loss='mae',
              optimizer='adam',
              metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(val_X, val_y), verbose=2, shuffle=False)
#score = model.evaluate(test_X,test_y)
# evaluate the model
_, train_acc = model.evaluate(train_X, train_y, verbose=0)
_, val_acc = model.evaluate(val_X, val_y, verbose=0)
print('Train: %.3f, Validation: %.3f' % (train_acc, val_acc))
#plot history
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='val')
pyplot.legend()
pyplot.show()

# Test set
# test file
file_test0 = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\output\\csv\\37_10.csv"
#test dataframe
df_test = pd.read_csv(file_test0,header=0,index_col=0) 
# Define variables 
df_test.replace(-1,np.NaN,inplace=True)
df_test.interpolate(inplace=True)
feat = df_test[df_test.columns[11:145]]
labels = df_test.mood.values
# Define X, y 
test_X = feat
test_y = labels
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
test_X = scaler.fit_transform(test_X)
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0],1, test_X.shape[1]))
# Evaluate model 
_, test_acc = model.evaluate(test_X, test_y, verbose=0)
print('Test: %.3f' % (test_acc))
