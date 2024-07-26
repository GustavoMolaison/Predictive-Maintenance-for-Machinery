import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read data
df_test1_fake = pd.read_csv('test_FD001.txt', delim_whitespace=True)
df_train1_fake = pd.read_csv('train_FD001.txt', delim_whitespace=True)
df_test1_fake.reset_index(drop=True, inplace=True)

# Function to modify DataFrame columns
def mod_df(df_var):
    column_names = ['unit number', 'time', 'operational setting 0', 'operational setting 1', 
                    'operational setting 2'] + [f'sensor measurement {i}' for i in range(0, 21)]
    new_df = df_var.copy()
    new_df.columns = column_names
    return new_df

# Apply column modifications
df_test1_processed = mod_df(df_test1_fake)
df_train1_processed = mod_df(df_train1_fake)

# Analyze the dataset
filt = (df_train1_processed['time'] == 1).sum()
print(f'train: {filt}')
filt = (df_test1_processed['time'] == 1).sum()
print(f'test: {filt}')

# Function to calculate time until breakdown
def count_time(df):
    time_break_list = df.groupby('unit number')['time'].max().tolist()
    return time_break_list

def time_info():
    mean_time = np.mean(count_time(df_train1_processed))
    breakout_time = count_time(df_train1_processed)
    print(f'Average time for machine to break down: {mean_time}')
    plt.bar(list(range(100)), breakout_time)
    plt.show()

# Calculate rolling statistics
def rolling_func(df):
    df_rol = pd.DataFrame()
    for i in range(3):
        df_rol[f'rolling_mean_oper {i}'] = df[f'operational setting {i}'].rolling(window=5).mean()
        df_rol[f'rolling_std_oper {i}'] = df[f'operational setting {i}'].rolling(window=5).std()
    for i in range(21):
        df_rol[f'rolling_mean_sensor {i}'] = df[f'sensor measurement {i}'].rolling(window=5).mean()
        df_rol[f'rolling_std_sensor {i}'] = df[f'sensor measurement {i}'].rolling(window=5).std()
    df = pd.concat([df, df_rol], axis=1)
    return df

# Apply rolling function
df_train1_processed = rolling_func(df_train1_processed)

# Function to generate lag features
def lag_feature(df, lags=[1, 2, 3]):
    dfs_lags = []
    for lag in lags:
        df_lag = pd.DataFrame()
        for i in range(3):
            df_lag[f'oper_set_lag {i}-{lag}'] = df[f'operational setting {i}'].shift(lag)
        for i in range(21):
            df_lag[f'sensor_measure_lag {i}-{lag}'] = df[f'sensor measurement {i}'].shift(lag)
        dfs_lags.append(df_lag)
    lag_features_concat = pd.concat(dfs_lags, axis=1)
    df = pd.concat([df, lag_features_concat], axis=1)
    df.bfill(inplace=True)         
    return df

# Apply lag features
df_train1_processed = lag_feature(df_train1_processed)

# Function to calculate usage duration features
def usage_dur_oper(df):
    df['oper_set_mean*'] = df[['operational setting 0', 'operational setting 1', 'operational setting 2']].mean(axis=1, skipna=True)
    df['sensor_mer_mean*'] = df[[f'sensor measurement {i}' for i in range(21)]].mean(axis=1, skipna=True)
    return df      

# Apply usage duration function
df_train1_processed = usage_dur_oper(df_train1_processed)

# Print some data to verify
print(df_train1_processed.head())