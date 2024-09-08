import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define a simple stateful LSTM model
def create_model(X_shape, y_shape):
    model = Sequential()
    model.add(LSTM(50, stateful=True, return_sequences=True, activation='relu'))
    model.add(Dense(1))  # Add a Dense layer with 1 output
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Example usage
X_shape = 10  # Number of time steps
y_shape = 20  # Number of features
model = create_model(X_shape, y_shape)
model.summary()