# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:25:15 2019

@author: melbs
"""
import pandas as pd
import numpy as np 
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
import sys 
sys.path.append("C:\\Users\\melbs\\OneDrive\\Desktop\\DSL")
from omid import *
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# %%
# Load and prepare dataset
# resample value
ir_val = '100ms'
# step size
n_steps = 20
#n_size = int(4000)
# Load and prepare dataset
df01= get_table(18,0, resample_interval= ir_val)#[:n_size]
df02 = get_table(26,0, resample_interval= ir_val)#[:n_size]
df03 = get_table(26,10, resample_interval= ir_val)#[:n_size]
df04 = get_table(27,0, resample_interval= ir_val)#[:n_size]
df05 = get_table(27,10, resample_interval= ir_val)#[:n_size]
df06 = get_table(34,10, resample_interval= ir_val)#[:n_size]
df07 = get_table(37,0, resample_interval= ir_val)#[:n_size]
df08 = get_table(37,10, resample_interval= ir_val)#[:n_size]
df09 = get_table(45,10, resample_interval= ir_val)#[:n_size]
df10 = get_table(49,0, resample_interval= ir_val)#[:n_size]
df11 = get_table(50,10, resample_interval= ir_val)#[:n_size]
df12 = get_table(51,0, resample_interval= ir_val)#[:n_size]
df13 = get_table(60,0, resample_interval= ir_val)#[:n_size]
df14 = get_table(60,10, resample_interval= ir_val)#[:n_size]
df15 = get_table(31,10, resample_interval= ir_val)#[:n_size]
df16 = get_table(31,0, resample_interval= ir_val)#[:n_size]
#%%
# concatenate datasize
dfs = [df01,df02,df03,df04,df05,df06,df07,df08,df09,df10,df11,df12,df13,df14,df15,df16]
#np.random.shuffle(dfs)
dataset = pd.concat(dfs,axis=0)
#scaled = make_scale(dataset)
# %%
#Calculate eye and mouth ratio 
dataset['eye_l_ratio'] = ratio_6(dataset,38,39,41,42,37,40) 
dataset['eye_r_ratio'] = ratio_6(dataset,44,45,47,48,43,46)
dataset['mouth_ratio'] = ratio_4(dataset,52,58,49,55)
#dataset['mouth_ratio'] = ratio_6(dataset,44,45,47,48,43,46)
# drop face columns 
dataset.drop(['face_x', 'face_y', 'face_w', 'face_h','time'], axis=1,inplace=True)
# save as float 32 and drop participant and mood columns
values = dataset
values = values.astype('float32')
values.drop(['mood','participant'],axis=1,inplace=True)
# Define X (features: x,y locations, and eye and mouth ratio)
X = values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
# define y (drowsy or alert)
y = dataset.mood[0:X.shape[0]].values
y[y!=0]=1
y = to_categorical(y)
# %%
# split into train and test sets 
# Use first 14 videos for training, and last two for testing 
train_size = int((dataset.shape[0]*(14/16)))
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
# design and train LSTM  network 
from keras.layers import Dropout
import timeit

start = timeit.default_timer()

model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
model.add(LSTM(100,return_sequences=False))
#model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0, shuffle=False)

stop = timeit.default_timer()

print('Time: ', stop - start)  
# %%
# plot loss and accuracy history 
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.show()

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, val_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Testing: %.3f' % (train_acc, val_acc))
#plot history
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.xlabel('epochs')
pyplot.ylabel('accuracy')
pyplot.legend()
pyplot.show()

#%%
# save predictions and probabilities
# show auc curve
y_pred = model.predict(X_test)
prob = model.predict_proba(X_test)
# keep probabilities for drowsy outcome only 
prob = prob[:,1]
y_true = y_test[:,1]
# calculate scores
ffe_auc = roc_auc_score(y_true, prob)
# summarize scores
print('ROC AUC=%.3f' % (ffe_auc))
# calculate roc curves
fpr, tpr, _ = roc_curve(y_true, prob)
# plot the roc curve for the model
pyplot.plot(fpr, tpr, linestyle='-', label='FFE-LSTM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# %%
pyplot.plot(prob, label='y_pred')
pyplot.plot(y_true, label='y_true')
pyplot.legend()
pyplot.xlabel('Samples')
pyplot.ylabel('Prediction')
pyplot.show()

# %% 
# Save predictions to csv with participant, mood, EAR, MAR
y_pred = pd.DataFrame(y_pred)

test_set = dataset[['participant','mood','eye_l_ratio','eye_r_ratio','mouth_ratio']]
test_set = test_set[train_size:X.shape[0]]
predictions = test_set[::20]
predictions['alert'] = y_pred[[0]].values
predictions['drowsy'] = y_pred[[1]].values
predictions.to_csv('prediction.csv')

# %% 
#for j in y_pred:
#	if j[0]>j[1]:
#		print("Driver is alert with the confidence of",(j[0]*100),"%")
#	else:
#		print("Driver is drowsy with the confidence of",(j[1]*100),"%")
#		print("Sounding the alarm now....")
#		# duration = 10  # second
#		# freq = 440  # Hz
#		# os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
#		for i in range(5):
#			os.system('say "Wake up now"')
