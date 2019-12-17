#%%
from omid import *
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime

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
df1 = get_table(31, 0, start_time=0, stop_time=60, resample_interval=None)
df2 = get_table(31, 10, start_time=0, stop_time=60, resample_interval=None)
# df = pd.concat([df1, df2])
# df = pd.concat([df2])



# X1 = df1[['px_8','py_8',
#             'px_36','px_37','px_38','px_39','px_40','px_41','px_42','px_43','px_44','px_45','px_46',
#             'px_47','px_48','px_49','px_51','px_52','px_53','px_54','px_55','px_56','px_57','px_58',       
#             'py_36','py_37','py_38','py_39','py_40','py_41','py_42','py_43','py_44','py_45','py_46',
#             'py_47','py_48','py_49','py_51','py_52','py_53','py_54','py_55','py_56','py_57','py_58',       
#             'face_x', 'face_y', 'face_w', 'face_h']]
# X2 = df2[['px_8','py_8',
#             'px_36','px_37','px_38','px_39','px_40','px_41','px_42','px_43','px_44','px_45','px_46',
#             'px_47','px_48','px_49','px_51','px_52','px_53','px_54','px_55','px_56','px_57','px_58',       
#             'py_36','py_37','py_38','py_39','py_40','py_41','py_42','py_43','py_44','py_45','py_46',
#             'py_47','py_48','py_49','py_51','py_52','py_53','py_54','py_55','py_56','py_57','py_58',       
#             'face_x', 'face_y', 'face_w', 'face_h']]

# X1 = df1[['px_8','py_8',
#             'px_36','px_37','px_38','px_39','px_40','px_41','px_42','px_43','px_44','px_45','px_46',
#             'px_47','px_48','px_49','px_51','px_52','px_53','px_54','px_55','px_56','px_57','px_58',       
#             'py_36','py_37','py_38','py_39','py_40','py_41','py_42','py_43','py_44','py_45','py_46',
#             'py_47','py_48','py_49','py_51','py_52','py_53','py_54','py_55','py_56','py_57','py_58',       
#             'face_x', 'face_y', 'face_w', 'face_h']]
# X2 = df2[['px_8','py_8',
#             'px_36','px_37','px_38','px_39','px_40','px_41','px_42','px_43','px_44','px_45','px_46',
#             'px_47','px_48','px_49','px_51','px_52','px_53','px_54','px_55','px_56','px_57','px_58',       
#             'py_36','py_37','py_38','py_39','py_40','py_41','py_42','py_43','py_44','py_45','py_46',
#             'py_47','py_48','py_49','py_51','py_52','py_53','py_54','py_55','py_56','py_57','py_58',       
#             'face_x', 'face_y', 'face_w', 'face_h']]

X2 = df2[['px_38', 'px_39', 'px_41', 'px_42', 'px_37', 'px_40','py_38', 'py_39', 'py_41', 'py_42', 'py_37', 'py_40']]

# scaled_tupled_X = make_tuples(X1, scale=True)
# scaled_X = make_tuples(X1, scale=False)
# tuples_X = make_scale(X1)

# scaled.drop(['face_x', 'face_y', 'face_w', 'face_h'], axis=1,inplace=True)


Y = df1.mood

df1.mood[df1.mood != 0] = 1
# Y = pd.get_dummies(df.mood)

# %%


# X1['eye_l_ratio'] = ratio_6(X1,37, 38, 40, 41, 36, 39)
# X1['eye_r_ratio'] = ratio_6(X1,43, 44, 46, 47, 42, 45)
# X1['mouth_ratio'] = ratio_4(X1,51, 57, 48, 54)
# X2['eye_l_ratio'] = ratio_6(X2,37, 38, 40, 41, 36, 39)
# X2['eye_r_ratio'] = ratio_6(X2,43, 44, 46, 47, 42, 45)
# X2['mouth_ratio'] = ratio_4(X2,51, 57, 48, 54)

X2['eye_l_ratio'] = ratio_6(X2,38, 39, 41, 42, 37, 40)


# %%
# plt.plot((X1['eye_l_ratio'] - X1['eye_l_ratio'].mean()) / X1['eye_l_ratio'].max(), 'b-')
# plt.plot((X2['eye_l_ratio'] - X2['eye_l_ratio'].mean()) / X2['eye_l_ratio'].max(), 'r-')
# plt.plot(X2['eye_l_ratio'], 'b-')

# rolling_window = '100ms'
# blink_threshold = 0.22


# print('blinks X1 = ', X1[X1['eye_l_ratio'] < blink_threshold]['eye_l_ratio'].count())
# print('blinks X2 = ', X2[X2['eye_l_ratio'] < blink_threshold]['eye_l_ratio'].count())

# baseline_X1 = X1.between_time('00:00:05', '00:00:10')['eye_l_ratio'].mean()
# baseline_X2 = X2.between_time('00:00:05', '00:00:10')['eye_l_ratio'].mean()

# # Rebased
# plt.subplot(211)
# plt.plot(X1['eye_l_ratio']-baseline_X1, 'b-')
# plt.plot(X2['eye_l_ratio']-baseline_X2, 'r-')
# # Originals
# plt.subplot(212)
# plt.plot(X1['eye_l_ratio'], 'b-')
# plt.plot(X2['eye_l_ratio'], 'r-')

# from scipy.signal import argrelextrema
# order = 6
# c_max_index = argrelextrema(1/X1['eye_l_ratio'].values, np.greater, order=order)
# print('spike detection, X1: ', len(c_max_index[0]))
# c_max_index = argrelextrema(1/X2['eye_l_ratio'].values, np.greater, order=order)
# print('spike detection, X2: ', len(c_max_index[0]))


# plt.hlines(X1['eye_l_ratio'].mean(), X1.index.min(), X1.index.max())
# plt.hlines(X2['eye_l_ratio'].mean(), X2.index.min(), X2.index.max(), color='y')
# # Print blinksatio'] < blink_threshold]['eye_l_ratio'].count())

# # plt.plot(X2['eye_l_ratio'], 'r-')

plt.ylim(0,0.5)
plt.plot(X2['eye_l_ratio'][0:200])

plt.show()

# %%
