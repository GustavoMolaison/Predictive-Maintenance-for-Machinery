import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df_test1_fake = pd.read_csv('test_FD001.txt')
df_train1_fake = pd.read_csv('train_FD001.txt')

df_test1_fake.reset_index(drop=True, inplace=True)

df_names_dict = {'df_test1':'df_test1', 'df_train1':'df_train1'}



def mod_df(df_var, new_df_name):
        
             
        bef_split = str(df_var.columns)
        column_test1 = bef_split.split() 


        new_df_name = pd.DataFrame( {'unit number' : [column_test1[0].strip("Index(['")]} )
        new_df_name['time'] = column_test1[1]
        for i in range(0, 3):
           new_df_name[f'operational setting {i}'] = column_test1[i + 2]
        for i in range(0, 19):
           new_df_name[f'sensor measurement {i}'] = column_test1[i + 5]




        for i in range (3000):
           print(i)
           bef_split = df_var.iloc[i]
           rows_test1 = bef_split.iloc[0].split()

           df_temp = pd.DataFrame({'unit number' : [rows_test1[0]]})   
           
           df_temp['time'] =  rows_test1[1]
           for i in range(0, 3):
                  df_temp[f'operational setting {i}'] = rows_test1[i + 2]
           for i in range(0, 19):
                  df_temp[f'sensor measurement {i}'] = rows_test1[i + 5] 
           new_df_name = pd.concat([new_df_name, df_temp], ignore_index=True)             

        
        print(new_df_name)    

mod_df(df_test1_fake, df_names_dict['df_test1'])

# print(df_test1_fake.iloc[0])
# bef_split = df_test1_fake.iloc[0]
# rows_test1 = bef_split.iloc[0].split()
# print(rows_test1)


# 13095