#%%
from omid import *
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#%% Imports
#  import libraries 
# from numpy import mean
# from numpy import std
# from numpy import dstack
# from pandas import read_csv
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.layers import GRU
# from keras.layers import Embedding
# import keras.layers
# from keras.regularizers import L1L2
# from keras.utils import to_categorical
# from matplotlib import pyplot
# from math import sqrt
# from numpy import concatenate
# from pandas import DataFrame
# from pandas import concat
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split




#%% Playground
df1 = get_table(60, 0, start_time=15, stop_time=580, resample_interval='50ms')
df2 = get_table(60, 10, start_time=15, stop_time=580, resample_interval='50ms')

# df = pd.concat([df1, df2])
# df = pd.concat([df2])



X1 = df1[['px_9', 'px_37', 'px_38', 'px_39',
       'px_40', 'px_41', 'px_42', 'px_43', 'px_44', 'px_45', 'px_46',
       'px_47', 'px_48', 'px_49', 'px_50', 'px_51', 'px_52', 'px_53',
       'px_54', 'px_55', 'px_56', 'px_57', 'px_58', 'px_59', 'py_9',
       'py_37', 'py_38', 'py_39', 'py_40', 'py_41', 'py_42', 'py_43',
       'py_44', 'py_45', 'py_46', 'py_47', 'py_48', 'py_49', 'py_50',
       'py_51', 'py_52', 'py_53', 'py_54', 'py_55', 'py_56', 'py_57',
       'py_58', 'py_59',
       'face_x', 'face_y', 'face_w', 'face_h']]
X2 = df2[['px_9', 'px_37', 'px_38', 'px_39',
       'px_40', 'px_41', 'px_42', 'px_43', 'px_44', 'px_45', 'px_46',
       'px_47', 'px_48', 'px_49', 'px_50', 'px_51', 'px_52', 'px_53',
       'px_54', 'px_55', 'px_56', 'px_57', 'px_58', 'px_59', 'py_9',
       'py_37', 'py_38', 'py_39', 'py_40', 'py_41', 'py_42', 'py_43',
       'py_44', 'py_45', 'py_46', 'py_47', 'py_48', 'py_49', 'py_50',
       'py_51', 'py_52', 'py_53', 'py_54', 'py_55', 'py_56', 'py_57',
       'py_58', 'py_59',
       'face_x', 'face_y', 'face_w', 'face_h']]

    
scaled_tupled_X = make_tuples(X1, scale=True)
scaled_X = make_tuples(X1, scale=False)
tuples_X = make_scale(X1)

# scaled.drop(['face_x', 'face_y', 'face_w', 'face_h'], axis=1,inplace=True)


Y = df1.mood
df1.mood[df1.mood != 0] = 1
# Y = pd.get_dummies(df.mood)

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


X1['eye_l_ratio'] = ratio_6(X1,38,39,41,42,37,40)
X1['eye_r_ratio'] = ratio_6(X1,44,45,47,48,43,46)
X1['mouth_ratio'] = ratio_6(X1,44,45,47,48,43,46)
X2['eye_l_ratio'] = ratio_6(X2,38,39,41,42,37,40)
X2['eye_r_ratio'] = ratio_6(X2,44,45,47,48,43,46)
X2['mouth_ratio'] = ratio_6(X2,44,45,47,48,43,46)



# %%
# plt.plot((X1['eye_l_ratio'] - X1['eye_l_ratio'].mean()) / X1['eye_l_ratio'].max(), 'b-')
# plt.plot((X2['eye_l_ratio'] - X2['eye_l_ratio'].mean()) / X2['eye_l_ratio'].max(), 'r-')
plt.plot(X1['eye_l_ratio'], 'b-')
plt.plot(X2['eye_l_ratio'], 'r-')


plt.show()

# %%
