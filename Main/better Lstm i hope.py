import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from procesing import procces_df, count_time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import TimeSeriesSplit


if __name__ == '__main__':
   df_train1 = pd.read_csv('train_FD001.txt', delim_whitespace=True)
   df_train1 = procces_df(df_train1)

   df_test1 = pd.read_csv('test_FD001.txt', delim_whitespace=True)
   df_test1 = procces_df(df_test1)

   breakout_list = count_time(df_train1)



def pred_and_eve(model, test_list):
    
    y_pred_list = []

    for test in test_list:
            y_pred = model.predict(test)
            y_pred_list.append(np.mean(y_pred))

    return y_pred_list

def minmaxscaling_sequences(df):
     
     df = df.bfill()
    
     scaler = MinMaxScaler(feature_range=(0,1))
     features = df.columns.tolist()
     features.remove('unit number')
     features.remove('time')
     scaled_data = scaler.fit_transform(df[features])
     scaled_df_fake = pd.DataFrame(scaled_data)
     scaled_df = pd.concat([df['unit number'], df['time'],  scaled_df_fake], axis=1)
     scaled_df.columns = df.columns

     return  scaled_df

def engine_data_generator(df, test = False, batch_size =1, seq_len = 10, engines_num = 26, epochs = 10):
    for epoch in list(range(epochs)):

     units = list(range(1, engines_num))
     for i in units:

        df = df[df['unit number'].isin(units)]

    
        # number of batches in one engine
        max_batches = len(df['unit number'] == i) // (seq_len * batch_size)
        
        index_start = 0
     #    List for engine data each variable is one batch.
        batches = []
     #    List for disfunction time for all engines, each variable is the same. Their amount is definy by number of batches for current unit.
        labels = []
        for batch in list(range(max_batches)):
            index_end = index_start + seq_len
            df_batch = df[index_start: index_end]
            index_start = index_start + seq_len

            
            if test == False:
                label_y = (df['time'] == i).max()


            batch_data = np.array(df_batch).reshape((batch_size, seq_len, len(df.columns)))        
            label_data = np.array([[label_y] * seq_len]).reshape(batch_size, seq_len)
            

            yield batch_data, label_data
         
        statefull_model.reset_states()



def LSTM_model(X_shape, y_shape):
     # RELU
     model = Sequential()
     model.add(LSTM(50, activation = 'relu', batch_input_shape=(1, X_shape, y_shape), stateful = True))
     model.add(Dense(1))
     model.compile(optimizer ='adam', loss = 'mse')

     # LEAKYRELU
     # model = Sequential()
     # model.add(LSTM(50, input_shape=(X_shape, y_shape), return_sequences = False))
     # model.add(LeakyReLU(alpha=0.3))
     # model.add(Dense(1))
     # model.compile(optimizer ='adam', loss = 'mse')

     return model


def model_fun(df, df_test, range_var, type = 'stateful'):
   
   if type ==  'stateful': 
     global statefull_model
    
     statefull_model = LSTM_model(10, 148)
 
     epochs_num = 3
     for epoch in list(range(epochs_num)):
        statefull_model_history = statefull_model.fit(engine_data_generator(df,
                                                      test = False,
                                                      batch_size =1,
                                                      seq_len = 10,
                                                      engines_num = range_var,
                                                      epochs = epochs_num),
                                                     steps_per_epoch = 11688,
                                                     epochs = epochs_num)
 
     return statefull_model, statefull_model_history


def evaluate_LSTM_model(model, X_test):
     y_pred = model.predict(engine_data_generator(X_test,
                                                      test = False,
                                                      batch_size =1,
                                                      seq_len = 10,
                                                      epochs = 10))
    
     
    
     print(F'NIEWIARYGONDE{y_pred.shape}')
     
     fig, ax = plt.subplots()
     # Plot the bar
     
     ax.hist(y_pred)

     

     # Label the plot
     # ax.set_ylabel('Predicted Value')
     ax.set_title('Single Predicted Value')

     # Show the plot
     plt.show()

     
     stop = input('press anything to stop')


model, model_history = model_fun(df_train1, df_test1, range_var =25, type = 'stateful')

evaluate_LSTM_model(model = model, X_test = df_test1)