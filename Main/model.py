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





def custom_train_test_split(df, test_ratio):
     unique_units = df['unit number'].unique()
     num_test_units = int(len(unique_units) * test_ratio)

     num_test_units = min(num_test_units, len(unique_units))

     test_units = unique_units[:num_test_units]
     train_units = unique_units[num_test_units:]
     
     test = df[df['unit number'].isin(test_units)]
     train = df[df['unit number'].isin(train_units)]
     
     X_test_list, y_list_test = splitting(test, 'test')
     X_train, Y_train = splitting(train, 'train')
     print(f'y_test_list{y_list_test[0]}')

     return  X_train, X_test_list, Y_train, y_list_test


def splitting(df, type):
      
      if type == 'test2':


        
        X_test_list = []

        for unit in df['unit number'].unique():
             unit_data = df[df['unit number'] == unit]
             X_test_list.append(unit_data)
        return   X_test_list  

   
      X_list = []
      y_list = []
      num = 1

      for i in (df['unit number'].unique()):
                spectrum = breakout_list[i - 1] - np.random.randint(25, 80)
                filt = df['unit number'] == i
                df_filt = df[filt]
       

                x_todrop = df_filt['time'] <  spectrum
                y_todrop = df_filt['time'] >= spectrum
         

                x_filtered = df_filt[x_todrop]
                y_filtered = df_filt[y_todrop]

                y = y_filtered['time'].max()
                print(f'yyyyyyyyyyyyyyyy{y}')
                y_to_go = pd.Series([y] * x_filtered.shape[0], index=x_filtered.index)
                print(f'debugging y_to_go{y_to_go.max()}')

                X_list.append(x_filtered)
                y_list.append(y_to_go)
                print(num)
                num += 1
   
      X_comb = pd.concat(X_list, ignore_index=True)
      y_comb = pd.concat(y_list, ignore_index=True)

      if type == 'test':


        
        X_test_list = []

        for unit in X_comb['unit number'].unique():
             unit_data = X_comb[X_comb['unit number'] == unit]
             X_test_list.append(unit_data)

        print(f'debugging{y_list[0]}')
        y_list_mean = [(series.max()) for series in y_list]   
        print(f'y_list_mean{y_list_mean[0]}')   
        return X_test_list, y_list_mean 
      

      if type == 'train':
#        train
         return X_comb, y_comb

# if __name__ == '__main__':
# X, y = splitting(df_train1)

def pred_and_eve(model, test_list):
    
    y_pred_list = []

    for test in test_list:
            y_pred = model.predict(test)
            y_pred_list.append(np.mean(y_pred))

    return y_pred_list


def modeling(df, test):
    
    

    X_train, X_val_list, Y_train, Y_val_list = custom_train_test_split(df, 0.2)

    print(Y_val_list[0])
    
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)

    
    y_pred_list = pred_and_eve(model, X_val_list)
         
    print("Shapes of predictions:", [pred.shape for pred in y_pred_list])
    print("Shapes of true values:", [val.shape for val in Y_val_list])
    
    
    mse =  mean_squared_error(Y_val_list, y_pred_list)
    r2 = r2_score(Y_val_list, y_pred_list)


    plt.scatter(Y_val_list, y_pred_list)
    plt.ylabel('Predicted values')
    plt.xlabel('Actual labels')
    plt.show()
    stop = input('Press anything to end')





    # predicting test files without true values
#     X_test_list = splitting(test, 'test')
#     pred_lists = pred_and_eve(model, X_test_list)

   
#     breakouts = []
#     for listt in pred_lists:
#        breakoutime = listt[-1]
#        breakouts.append(breakoutime)
      
       
     


#    #  mse =  mean_squared_error(test, y_pred)
#    #  r2 = r2_score(test, y_pred)

#     unit_num = range(len(X_test_list))
#     plt.scatter(breakouts,  unit_num)
#     plt.show()
#     stop = input('Press anything to end')

modeling(df = df_train1, test = df_test1)