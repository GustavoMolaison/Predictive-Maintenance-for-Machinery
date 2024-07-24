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





def mod_df(df_var):
        
             
        column_names = ['unit number', 'time', 'operational setting 0', 'operational setting 1', 
                     'operational setting 2'] + [f'sensor measurement {i}' for i in range(0, 19)]


        new_df = pd.DataFrame(columns = column_names)

        new_df = df_var.copy()
        new_df.columns = column_names
        

        return new_df

         

mod_df(df_test1_fake)

df_test1_processed = mod_df(df_test1_fake)
df_train1_processed = mod_df(df_train1_fake)


print(df_test1_processed.head())
print(df_train1_processed.head())
         