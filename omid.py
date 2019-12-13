#%% Importsface_points_to_keep

import pandas as pd
import numpy as np 
from pandas import read_csv
import glob
import os
import logging
import re


#%% Variables

# Code Tune
logging.getLogger().setLevel(logging.INFO)

csv_pick_regex= r'6*_[5,0].csv'
csv_pick_regex= r'6*.csv'


# Table Tune
face_points_to_keep = []
face_points_to_keep += [9]                     # Nose
face_points_to_keep += [37,38,39,40,41,42]     # Left Eye
face_points_to_keep += [43,44,45,46,47,48]     # Right Eye
face_points_to_keep += [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59] # Outer Lip
# face_points_to_keep += [18, 19, 20, 21, 22, 23, 24, 25, 26] # Eyebrows

# time_min = 60
# time_max = 360
# interval = '1s'



columns_to_keep = ['participant', 'mood', 'time'] + \
                    [f'px_{x}' for x in face_points_to_keep] + \
                    [f'py_{x}' for x in face_points_to_keep] +\
                    ['face_x','face_y','face_w','face_h']



# filesDepth3 = glob.glob(os.path.join('output','csv',csv_pick_regex)) 

# # Checks
# logging.info(f"face points={face_points_to_keep}")
# logging.info(f"Processing files:")
# for x in filesDepth3:
#     logging.info(f"  {x}")

# #%% Process Tables
# # Find Tables

# #Load Tables
# input_tables = {}
# for table_path in filesDepth3:
#     # Load
#     table = pd.read_csv(table_path)
    
#     # Resample time
#     table['date'] = pd.to_datetime(table.time, unit='s')
#     table = table.resample(interval, on = 'date').mean()
    
#     # Drop columns we don't need
#     table = table.filter(columns_to_keep)

#     # Fill missing data
#     table.replace(-1, np.NaN, inplace=True)
#     table.interpolate(inplace=True)

#     input_tables[os.path.basename(table_path)[:-4]] = table


# # Merge Tables
# merged_table = pd.concat(input_tables.values())

# # # Drop columns we don't need
# # merged_table = merged_table.filter(columns_to_keep)



# # Fix Data Types
# merged_table[['participant', 'mood']] = merged_table[['participant', 'mood']].astype('int32')
# pxy_cols = [x for x in merged_table.columns if re.compile('p[xy]_*').match(x)]
# merged_table[pxy_cols] = merged_table[pxy_cols].astype('int32')

# # Trim head and tail of the video
# merged_table.drop(merged_table[ merged_table['time'] > time_max ].index, inplace=True)
# merged_table.drop(merged_table[ merged_table['time'] < time_min ].index, inplace=True)



# %%
def get_table(participant, mood, start_time=60, stop_time=360, resample_interval='100ms', 
                base_path=None):
    
    # Find File
    if base_path is None:
        base = os.path.join('output','csv')
    else:
        base = base_path
    
    files = glob.glob(os.path.join(base, f'{participant}_{mood}.csv'))
    
    if(len(files) !=1 ):
        logging.error(f"Looked for {participant}_{mood}.csv and found {len(files)} tables. Need to match with one table only.")
        raise RuntimeError

    # Load
    logging.info(f"loading {files[0]}")
    table = pd.read_csv(files[0])
    
    # Resample time
    table['date'] = pd.to_datetime(table.time, unit='s')
    table = table.resample(resample_interval, on = 'date').mean()
    
    # Drop columns we don't need
    table = table.filter(columns_to_keep)

    # Fill missing data
    table.replace(-1, np.NaN, inplace=True)
    table.interpolate(inplace=True)

    # Fix Data Types
    table[['participant', 'mood']] = table[['participant', 'mood']].astype('int32')
    pxy_cols = [x for x in table.columns if re.compile('p[xy]_*').match(x)]
    table[pxy_cols] = table[pxy_cols].astype('int32')

    # Trim head and tail of the video
    table.drop(table[ table['time'] > stop_time ].index, inplace=True)
    table.drop(table[ table['time'] < start_time ].index, inplace=True)

    return table




# %%
def make_tuples(df_in, scale=False):
    print("Potato2")
    return_df = pd.DataFrame()

    if scale is False:
        scaled_df = df_in
    else:
        scaled_df = pd.DataFrame()
        for col in df_in.columns.values:
            if 'px_' in col:
                scaled_df[col] = df_in[col].sub(df_in['face_x']).div(df_in['face_w'])
            elif 'py_' in col:
                scaled_df[col] = df_in['face_y'].sub(df_in[col]).div(df_in['face_h'])

    for col in scaled_df.columns.values:
        if('px_' in col):
            idx = col[3:]
            return_df[f'p{idx}'] = scaled_df[[f'px_{idx}', f'py_{idx}']].apply(tuple, axis=1)
    
    return return_df