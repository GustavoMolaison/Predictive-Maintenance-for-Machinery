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
# print(df_test1_processed.head())
# print(df_train1_processed.head())
         