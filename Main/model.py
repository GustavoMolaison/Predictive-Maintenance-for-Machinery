import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from procesing import procces_df, count_time
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == '__main__':
   df_train1 = pd.read_csv('train_FD001.txt', delim_whitespace=True)
   df_train1 = procces_df(df_train1)

   df_test1 = pd.read_csv('test_FD001.txt', delim_whitespace=True)
   df_test1 = procces_df(df_test1)

   breakout_list = count_time(df_train1)


def splitting(df):
   X_list = []
   y_list = []

   for i in range (1, 100):
       spectrum = breakout_list[i - 1] - np.random.randint(25, 80)
       filt = df['unit number'] == i
       df_filt = df[filt]
       

       x_todrop = df_filt['time'] < spectrum
       y_todrop = df_filt['time'] >=spectrum
         

       x_filtered = df_filt[x_todrop]
       y_filtered = df_filt[y_todrop]

       y_todrop2 = y_filtered['time'] == y_filtered['time'].max()
       y_filt2 =  y_filtered[y_todrop2]

          

       X_list.append(x_filtered)
    #    y_list.append(y_todrop)
       y_list.append(y_filt2)
   
    #    X_comb = pd.concat(X_list, ignore_index=True)
    #    y_comb = pd.concat(y_list, ignore_index=True)
   print((X_list[0]))
   print(f'{y_list[0]}')
   return X_list, y_list

# if __name__ == '__main__':
# X, y = splitting(df_train1)




def modeling(train, test):
    
    X, y = splitting(train)

    model = RandomForestRegressor()
    model.fit(X[0], y[0])
    
    y_pred = model.predict(test)
    mse =  mean_squared_error(test, y_pred)
    r2 = r2_score(test, y_pred)


    plt.scatter(test, y_pred)

modeling(train = df_train1, test = df_test1)