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


def splitting(df, type):

   if type == 'test':
        units = df['unit number'].unique()
        X_test_list = []

        for unit in units:
             unit_data = df[df['unit number'] == unit]
             X_test_list.append(unit_data)

        return X_test_list 



   if type == 'train':
      X_list = []
      y_list = []
      num = 1

      for i in range (len(df['unit number'].unique() + 1)):
                spectrum = breakout_list[i - 1] - np.random.randint(25, 80)
                filt = df['unit number'] == i
                df_filt = df[filt]
       

                x_todrop = df_filt['time'] <  spectrum
                y_todrop = df_filt['time'] >= spectrum
         

                x_filtered = df_filt[x_todrop]
                y_filtered = df_filt[y_todrop]

                y = y_filtered['time'].max()
      

                y_to_go = pd.Series([y] * x_filtered.shape[0], index=x_filtered.index)


                X_list.append(x_filtered)
                y_list.append(y_to_go)
                print(num)
                num += 1
   
      X_comb = pd.concat(X_list, ignore_index=True)
      y_comb = pd.concat(y_list, ignore_index=True)
   
      return X_comb, y_comb

# if __name__ == '__main__':
# X, y = splitting(df_train1)

def pred_and_eve(model, test_list):
    
    pred_list = []

    for test in test_list:
            y_pred = model.predict(test)
            pred_list.append(y_pred)

    return pred_list


def modeling(df, test):
    
    X, y = splitting(df, 'train')

    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state = 42)

    model = RandomForestRegressor()
    model.fit(X, y)
    
    X_test_list = splitting(test, 'test')
    pred_lists = pred_and_eve(model, X_test_list)

   
    breakouts = []
    for listt in pred_lists:
       breakoutime = listt[-1]
       breakouts.append(breakoutime)
      
       
     


   #  mse =  mean_squared_error(test, y_pred)
   #  r2 = r2_score(test, y_pred)

    unit_num = range(len(X_test_list))
    plt.scatter(breakouts,  unit_num)
    plt.show()
    stop = input('Press anything to end')

modeling(df = df_train1, test = df_test1)