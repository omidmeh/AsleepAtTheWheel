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
from omid import get_table


# convert series to supervised learning
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

# Load and prepare dataset
df1 = get_table(31,0)
df2 = get_table(31,10)
df3 = get_table(45,0)
df4 = get_table(45,10)
df5 = get_table(51,0)
df6 = get_table(51,10)
df7 = get_table(60,0)
df8 = get_table(60,10)


dataset = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],axis=0)

# specify number of timesteps 
n_steps = 5 
n_features = 48
# frame as supervised learning
values = dataset[dataset.columns[3:51]]

values = values.astype('float32')
reframed = series_to_supervised(values, n_steps, 1)
print(reframed.shape)

# reshape dataframe into (x,y) coordinate tuples
#feat_tuple = pd.DataFrame()
#i=0
#while i in range(0,528):
#    feat_tuple[i]=(reframed[[reframed.columns[i],reframed.columns[i+1]]].apply(tuple,axis=1))
#    i = i+2
    
X = array(reframed)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = dataset.mood[0:23995].values
y[y!=0]=1
y = to_categorical(y)
# split into train and test sets
X_train = X[0:18000] 
X_test = X[18000:23995]
y_train = y[0:18000] 
y_test = y[18000:23995]

# Reshape input to 3D 
X_train = X_train.reshape(3600,5,288)
y_train = y_train[::5]
X_test = X_test.reshape(1199,5,288)
y_test = y_test[::5]

# design network 
# tune learning parameters 
# tune optimizers  (adam)
# tune lstm input 
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(LSTM(50))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, val_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Validation: %.3f' % (train_acc, val_acc))
#plot history
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='val')
pyplot.legend()
pyplot.show()

y_pred = model.predict(X_test)
pyplot.plot(y_pred[0])
pyplot.plot(y[0])
pyplot.show()