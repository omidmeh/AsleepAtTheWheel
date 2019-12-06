# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:56:25 2019

@author: melbs
"""

# 
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
file = "C:\\Users\\melbs\\OneDrive\\Desktop\\DSL\\facial-landmarks\\33_0.csv"
df = pd.read_csv(file,header=0,index_col=0) 
 
# Define variables 
time = df.time
dt = time.diff(periods=1).fillna(0.).values
feat = df[df.columns[9:145]]
labels = df.mood.values

dist = feat.diff(periods=1).fillna(0.)
dist = dist.add_suffix('_dist')
speed = feat.diff(periods=1).fillna(0.)/dt[1]
speed = speed.add_suffix('_speed')
accel = feat.diff(periods=1).fillna(0.)/dt[1]/dt[1]
accel = accel.add_suffix('_accel')

#
X = pd.concat([dist,speed,accel],axis=1).values
y = labels 

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Split the dataset in two equal parts
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=0)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
y_pred = model.predict(test_X)
# calculate RMSE
rmse = sqrt(mean_squared_error(test_y, y_pred))
print('Test RMSE: %.3f' % rmse)
