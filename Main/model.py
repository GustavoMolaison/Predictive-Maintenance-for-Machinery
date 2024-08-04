import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from procesing import procces_df, count_time
from sklearn.metrics import mean_squared_error, r2_score

# Entry point of the script
if __name__ == '__main__':
    # Load and preprocess the training data
    df_train1 = pd.read_csv('train_FD001.txt', delim_whitespace=True)
    df_train1 = procces_df(df_train1)

    # Load and preprocess the test data
    df_test1 = pd.read_csv('test_FD001.txt', delim_whitespace=True)
    df_test1 = procces_df(df_test1)

    # Create a list of breakpoints for the training data
    breakout_list = count_time(df_train1)

# Custom train-test split function
def custom_train_test_split(df, test_ratio):
    unique_units = df['unit number'].unique()
    num_test_units = int(len(unique_units) * test_ratio)

    num_test_units = min(num_test_units, len(unique_units))

    # Split units into test and train
    test_units = unique_units[:num_test_units]
    train_units = unique_units[num_test_units:]
    
    test = df[df['unit number'].isin(test_units)]
    train = df[df['unit number'].isin(train_units)]
    
    # Split data into features and targets
    X_test_list, y_list_test = splitting(test, 'test')
    X_train, Y_train = splitting(train, 'train')

    print(f'y_test_list{y_list_test[0]}')

    return X_train, X_test_list, Y_train, y_list_test

# Function to split data based on 'train' or 'test' type
def splitting(df, type):
    if type == 'test2':
        X_test_list = []
        for unit in df['unit number'].unique():
            unit_data = df[df['unit number'] == unit]
            X_test_list.append(unit_data)
        return X_test_list  

    X_list = []
    y_list = []
    num = 1

    for i in df['unit number'].unique():
        spectrum = breakout_list[i - 1] - np.random.randint(25, 80)
        filt = df['unit number'] == i
        df_filt = df[filt]

        # Filter out the data based on time and spectrum
        x_todrop = df_filt['time'] < spectrum
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
        return X_comb, y_comb

# Function to predict and evaluate
def pred_and_eve(model, test_list):
    y_pred_list = []

    for test in test_list:
        y_pred = model.predict(test)
        y_pred_list.append(np.mean(y_pred))

    return y_pred_list

# Function to train and evaluate the model
def modeling(df, test):
    # Hyperparameters for RandomizedSearchCV
    tunning_set = {'n_estimators': list(range(1, 101, 25)),
                   'max_depth': list(range(20, 100, 10)),
                   'min_samples_split': list(range(2, 26, 2))}

    X_train, X_val_list, Y_train, Y_val_list = custom_train_test_split(df, 0.2)
    print(Y_val_list[0])

    # Initialize and tune the RandomForestRegressor
    model = RandomForestRegressor()
    tunned_model = RandomizedSearchCV(estimator=model, param_distributions=tunning_set, 
                                       n_jobs=-1, n_iter=5, scoring='neg_mean_squared_error', 
                                       cv=5, random_state=42)
    tunned_model.fit(X_train, Y_train)
    best_params = tunned_model.best_params_
    best_score = tunned_model.best_score_

    # Predict and evaluate
    y_pred_list = pred_and_eve(tunned_model, X_val_list)
    mse = mean_squared_error(Y_val_list, y_pred_list)
    r2 = r2_score(Y_val_list, y_pred_list)

    # Plot results
    plt.scatter(Y_val_list, y_pred_list)
    plt.ylabel('Predicted values')
    plt.xlabel('Actual labels')
    plt.show()
    print(f'mean_squared_error; {mse}')
    print(f'r2_score {r2}')
    stop = input('Press anything to end')

# Function to compare basic and tuned models
def modeling_compare(df, test=None):
    # Basic model
    X_train, X_val_list, Y_train, Y_val_list = custom_train_test_split(df, 0.2)
    print(Y_val_list[0])
    
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)

    y_pred_list_basic = pred_and_eve(model, X_val_list)
    mse_basic = mean_squared_error(Y_val_list, y_pred_list_basic)
    r2_basic = r2_score(Y_val_list, y_pred_list_basic)

    # Plot basic model results
    fig_basic, ax_basic = plt.subplots()
    ax_basic.scatter(Y_val_list, y_pred_list_basic)
    ax_basic.set_ylabel('Predicted values (basic model)')
    ax_basic.set_xlabel('Actual labels')
    ax_basic.set_title('Basic Model')
    
    print(f'mean_squared_error; {mse_basic}')
    print(f'r2_score {r2_basic}')

    # Tuned model
    tunning_set = {'n_estimators': list(range(1, 101, 25)),
                   'max_depth': list(range(20, 100, 10)),
                   'min_samples_split': list(range(2, 26, 2))}

    X_train, X_val_list, Y_train, Y_val_list = custom_train_test_split(df, 0.2)
    print(Y_val_list[0])
    
    model = RandomForestRegressor()
    tunned_model = RandomizedSearchCV(estimator=model, param_distributions=tunning_set, 
                                       n_jobs=-1, n_iter=5, scoring='neg_mean_squared_error', 
                                       cv=5, random_state=42)
    tunned_model.fit(X_train, Y_train)
    best_params = tunned_model.best_params_
    best_score = tunned_model.best_score_

    y_pred_list = pred_and_eve(tunned_model, X_val_list)
    mse = mean_squared_error(Y_val_list, y_pred_list)
    r2 = r2_score(Y_val_list, y_pred_list)

    # Plot tuned model results
    fig_tunned, ax_tunned = plt.subplots()
    ax_tunned.scatter(Y_val_list, y_pred_list)
    ax_tunned.set_ylabel('Predicted values (tuned model)')
    ax_tunned.set_xlabel('Actual labels')
    ax_tunned.set_title('Tuned Model')
    
    print(f'mean_squared_error; {mse}')
    print(f'r2_score {r2}')
    
    return ax_basic, ax_tunned

# Function to compare basic and tuned models and display them side by side
def compare(df):
    ax_basic, ax_tunned = modeling_compare(df)

    # Create a figure with 1 row and 2 columns of subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot basic model scatter
    axs[0].scatter(ax_basic.collections[0].get_offsets()[:, 0], 
                   ax_basic.collections[0].get_offsets()[:, 1])
    axs[0].set_title('Basic Model')

    # Plot tuned model scatter
    axs[1].scatter(ax_tunned.collections[0].get_offsets()[:, 0], 
                   ax_tunned.collections[0].get_offsets()[:, 1])
    axs[1].set_title('Tuned Model')

    # Set main title and show plots
    fig.suptitle('Model Comparison')
    plt.tight_layout()
    plt.show()
    stop = input('Press anything to end')

# Call the compare function with the training data
compare(df=df_train1)








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