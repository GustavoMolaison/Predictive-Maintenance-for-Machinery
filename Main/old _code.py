import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df_test1_fake = pd.read_csv('test_FD001.txt', delim_whitespace=True)
df_train1_fake = pd.read_csv('train_FD001.txt', delim_whitespace=True)

df_test1_fake.reset_index(drop=True, inplace=True)



# print(len(df_test1_fake.columns))

def mod_df(df_var):
        
             
        column_names = ['unit number', 'time', 'operational setting 0', 'operational setting 1', 
                     'operational setting 2'] + [f'sensor measurement {i}' for i in range(0, 21)]


        new_df = pd.DataFrame(columns = column_names)

        new_df = df_var.copy()
        new_df.columns = column_names
        

        return new_df

         



df_test1_processed = mod_df(df_test1_fake)
df_train1_processed = mod_df(df_train1_fake)

# Analyzing the dataset
filt = (df_train1_processed['time'] == 1).sum()
print(f'train: {filt}')

filt = (df_test1_processed['time'] == 1).sum()
print(f'test: {filt}')


def count_time(df):

        # original worst version

        # time_break_list = []
        # for j in range (100):
        #   time_bef_break = 1
        #   filt = (df_train1_processed['time'] != 1)
        #   # print(filt)
        #   for i in filt:
        #         if i == True:
        #                 time_bef_break += 1
        #         if i == False:
        #                 break        
        #   time_break_list.append(time_bef_break)
        # return np.mean(time_break_list) 

        # better version

        time_break_list = df.groupby('unit number')['time'].max().tolist()
        
        
        return (time_break_list)



def time_info():
         mean_time = np.mean(count_time(df_train1_processed))
         breakout_time = count_time(df_train1_processed)
         print(f'averegae time for machine to break down its: {mean_time}')

         plt.bar(list(range(100)), breakout_time)
         plt.show()

# time_info()
# print(df_test1_processed.head())
# print(df_train1_processed.head())
         
def rolling_func(df):
         df_rol = pd.DataFrame()
         for i in range (3):
                 df_rol[f'rolling_mean_oper {i}'] = df[f'operational setting {i}'].rolling(window=5).mean()
                 df_rol[f'rolling_std_oper {i}'] =  df[f'operational setting {i}'].rolling(window=5).std()
         for i in range (21):
                    df_rol[f'rolling_mean_sensor {i}'] =  df[f'sensor measurement {i}'].rolling(window=5).mean()
                    df_rol[f'rolling_std_sensor {i}'] =  df[f'sensor measurement {i}'].rolling(window=5).std()
        #  print(df_rol['rolling_mean_oper 0'])
         df = pd.concat([df, df_rol], axis = 1)
        #  print(df['rolling_mean_oper 0'])
        #  oper_setings = [df['rolling_mean_oper 0'] ,df['rolling_mean_oper 1'], df['rolling_mean_oper 2'] ]
        #  print(oper_setings)
         return df
         

df_train1_processed = rolling_func(df_train1_processed)         

def lag_feature(df, lags =[1,2,3]):
        # for i in range(3):
        #         df[f'oper_set_lag {i}-1'] = df[f'operational setting {i}'].shift(1)
        #         df[f'oper_set_lag {i}-2'] = df[f'operational setting {i}'].shift(2) 
        #         df[f'oper_set_lag {i}-3'] = df[f'operational setting {i}'].shift(3)
        # for i in range(21):
        #         df[f'sensor_measure_lag {i}-1'] = df[f'sensor measurement {i}'].shift(1)
        #         df[f'sensor_measure_lag {i}-2'] = df[f'sensor measurement {i}'].shift(2)  
        #         df[f'sensor_measure_lag {i}-3'] = df[f'sensor measurement {i}'].shift(3)   



        # for i in range(3):
        #    for lag in range (1, lags + 1):
        #            df[f'oper_set_lag {i}-{lag}'] = df[f'operational setting {i}'].shift(lag)
        # for i in range(21):
        #    for lag in range (1, lags + 1):
        #            df[f'sensor_measure_lag {i}-{lag}'] = df[f'sensor measurement {i}'].shift(lag)   
        # df.fillna(method = 'bfill', inplace = True)     
        # 
        dfs_lags = []
        for lag in lags:
                df_lag = pd.DataFrame()
                for i in range(3):
                      df_lag[f'oper_set_lag {i}-{lag}'] = df[f'operational setting {i}'].shift(lag)  
                for i in range(21):
                      df_lag[f'sensor_measure_lag {i}-{lag}'] = df[f'sensor measurement {i}'].shift(lag)
                dfs_lags.append(df_lag)
        lag_featuers_concat = pd.concat(dfs_lags, axis = 1)
        df = pd.concat([df, lag_featuers_concat], axis = 1)

        df.bfill( inplace = True)         
        
        return df

df_train1_processed = lag_feature(df_train1_processed)



def means(df):
#     rolling_func(df)

      df['oper_set_mean*'] = df[['operational setting 0', 'operational setting 1', 'operational setting 2']].mean(axis=1, skipna = True)
      

      

      df['sensor_mer_mean*'] = df[[f'sensor measurement {i}' for i in range (21) ]].mean(axis =1, skipna = True)   

      return df      
       
# df_train1_processed = rolling_func(df_train1_processed)
df_train1_processed = means(df_train1_processed)

