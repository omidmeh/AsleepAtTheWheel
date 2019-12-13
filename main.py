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
df1 = get_table(45, 0)
df1 = get_table(60, 0)
df2 = get_table(45, 10)
df2 = get_table(60, 10)

# df = pd.concat([df1, df2])
df = pd.concat([df2])



X = df[['px_9', 'px_37', 'px_38', 'px_39',
       'px_40', 'px_41', 'px_42', 'px_43', 'px_44', 'px_45', 'px_46',
       'px_47', 'px_48', 'px_49', 'px_50', 'px_51', 'px_52', 'px_53',
       'px_54', 'px_55', 'px_56', 'px_57', 'px_58', 'px_59', 'py_9',
       'py_37', 'py_38', 'py_39', 'py_40', 'py_41', 'py_42', 'py_43',
       'py_44', 'py_45', 'py_46', 'py_47', 'py_48', 'py_49', 'py_50',
       'py_51', 'py_52', 'py_53', 'py_54', 'py_55', 'py_56', 'py_57',
       'py_58', 'py_59',
       'face_x', 'face_y', 'face_w', 'face_h']]

    
# scaled_X = make_tuples(X, scale=True)
# scaled.drop(['face_x', 'face_y', 'face_w', 'face_h'], axis=1,inplace=True)


Y = df.mood
df.mood[df.mood != 0] = 1
# Y = pd.get_dummies(df.mood)

# %%
def dist(mx, my ,nx, ny):
    return np.sqrt(np.square(mx-nx) + np.square(my-ny))

def mid(x1, x2):
    return (x1+x2)/2

def ratio_6(t1,t2,b1,b2,l,r):
    x1_m= mid(X[f'px_{t1}'], X[f'px_{t2}'])
    y1_m = mid(X[f'py_{t1}'], X[f'py_{t2}'])
    x2_m = mid(X[f'px_{b1}'], X[f'px_{b2}'])
    y2_m = mid(X[f'py_{b1}'], X[f'py_{b2}'])

    return dist(x1_m,y1_m,x2_m,y2_m) / dist(X[f'px_{l}'], X[f'py_{l}'], X[f'px_{r}'], X[f'py_{r}'])


X['eye_l_ratio'] = ratio_6(38,39,41,42,37,40)
X['eye_r_ratio'] = ratio_6(44,45,47,48,43,46)
X['mouth_ratio'] = ratio_6(44,45,47,48,43,46)



# %%
plt.plot(X['eye_l_ratio'])
plt.plot(X['eye_l_ratio'])

plt.show()

# %%
