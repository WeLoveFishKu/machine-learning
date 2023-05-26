import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from Model import model
from ProcessData import processdata

model_functions = model()
processdata_functions = processdata()

data = pd.read_csv('FishPrice.csv', parse_dates=['Date']).drop(columns='Unnamed: 0')
scaled_data, store_scalers = processdata_functions.scale_data(data)

split_index = 100
provinces = scaled_data['Provinsi'].unique()
train_price, train_date, test_price, test_date = np.array([]), np.array([], dtype='datetime64'), np.array([]), np.array([], dtype='datetime64')

for province in provinces:
    data_provinsi = scaled_data.loc[scaled_data['Provinsi'] == province, ['Date', 'Ikan Bandeng']]
    price_data = data_provinsi['Ikan Bandeng'].values
    date_data = data_provinsi['Date'].values
    train_price_, train_date_, test_price_, test_date_ = model_functions.prepare_dataset(price_data, date_data, split_index)
    train_price = np.concatenate((train_price, train_price_))
    train_date = np.concatenate((train_date, train_date_))
    test_price = np.concatenate((test_price, test_price_))
    test_date = np.concatenate((test_date, test_date_))

# Parameters
window_size = 7
batch_size = 16
shuffle_buffer_size = 1000

# Generate the dataset windows
# train_dataset = model_functions.windowed_dataset(train_price, window_size, batch_size, shuffle_buffer_size)
# train_dataset_ = model_functions.windowed_dataset(train_price_, window_size, batch_size, shuffle_buffer_size)

# # Build the Model
# ml_model = model_functions.create_model(window_size)

# # Find best learning rate for the model
# _ = model_functions.find_best_learningrate(ml_model, train_dataset)

# # # Set the learning rate
# learning_rate = 4e-4

# # Set the number of epochs
# epochs = 2000

# # Train the model
# ml_model = model_functions.model_fitting(ml_model, learning_rate, epochs, train_dataset)

# # Plot predictions on train dataset
# _ = model_functions.predict_using_trained_data(ml_model, train_dataset_, train_price_)