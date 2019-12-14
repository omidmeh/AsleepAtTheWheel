# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:25:15 2019

@author: melbs
"""
import pandas as pd
import numpy as np 
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from numpy import split
from numpy import array
import sys 
from sklearn.model_selection import train_test_split
sys.path.append("C:\\Users\\melbs\\OneDrive\\Desktop\\DSL")
from omid import *


# %% convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# %%
# Load and prepare dataset
ir_val = '50ms'
df1 = get_table(31,0, resample_interval= ir_val)
df2 = get_table(31,10, resample_interval= ir_val)
df3 = get_table(45,0, resample_interval= ir_val)
df4 = get_table(45,10, resample_interval= ir_val)
df5 = get_table(51,0, resample_interval= ir_val)
df6 = get_table(51,10, resample_interval= ir_val)
df7 = get_table(60,0, resample_interval= ir_val)
df8 = get_table(60,10, resample_interval= ir_val)
#%%
dataset = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],axis=0)
#scaled = make_scale(dataset)
# %%
def dist(mx, my ,nx, ny):
    return np.sqrt(np.square(mx-nx) + np.square(my-ny))

def mid(x1, x2):
    return (x1+x2)/2

def ratio_6(table, t1,t2,b1,b2,l,r):
    x1_m= mid(table[f'px_{t1}'], table[f'px_{t2}'])
    y1_m = mid(table[f'py_{t1}'], table[f'py_{t2}'])
    x2_m = mid(table[f'px_{b1}'], table[f'px_{b2}'])
    y2_m = mid(table[f'py_{b1}'], table[f'py_{b2}'])

    return dist(x1_m,y1_m,x2_m,y2_m) / dist(table[f'px_{l}'], table[f'py_{l}'], table[f'px_{r}'], table[f'py_{r}'])


dataset['eye_l_ratio'] = ratio_6(dataset,38,39,41,42,37,40)
dataset['eye_r_ratio'] = ratio_6(dataset,44,45,47,48,43,46)
dataset['mouth_ratio'] = ratio_6(dataset,44,45,47,48,43,46)
# %% 
# drop face columns 
dataset.drop(['face_x', 'face_y', 'face_w', 'face_h','time','participant'], axis=1,inplace=True)
#dataset = make_xy(dataset)
# %%
# specify number of timesteps 
n_steps = 5
n_features = dataset.shape[1]-1
# frame as supervised learning
values = dataset
values = values.astype('float32')
values.drop(['mood'],axis=1,inplace=True)
# %%
X = values
reframed = series_to_supervised(X, n_steps, 1)
print(reframed.shape)    
X = array(reframed)
# %% normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
# define y 
y = dataset.mood[0:X.shape[0]].values
y[y!=0]=1
y = to_categorical(y)
# %%
# split into train and test sets
train_size = int(dataset.shape[0]*0.75-10)
X_train = X[0:train_size] 
X_test = X[train_size:X.shape[0]]
y_train = y[0:train_size] 
y_test = y[train_size:X.shape[0]]
# %%
# Reshape input to 3D 
train_resize = int(X_train.shape[0]/n_steps)
test_resize = int(X_test.shape[0]/n_steps)

X_train = X_train.reshape(train_resize,n_steps,X.shape[1])
y_train = y_train[::n_steps]
X_test = X_test.reshape(test_resize,n_steps,X.shape[1])
y_test = y_test[::n_steps]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#%% 
# design network 
# tune learning parameters 
# tune optimizers  (adam)
# tune lstm input 
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=False))
#model.add(LSTM(50,return_sequences=True))
#model.add(LSTM(50))
#model.add(LSTM(50))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0, shuffle=False)
# %%
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('hidden units')
pyplot.ylabel('loss')
pyplot.show()

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, val_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Validation: %.3f' % (train_acc, val_acc))
#plot history
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='val')
pyplot.xlabel('hidden units')
pyplot.ylabel('accuracy')
pyplot.legend()
pyplot.show()

#y_pred = model.predict(X_test)
#pyplot.plot(y_pred[0])
#pyplot.plot(y[0])
#pyplot.show()