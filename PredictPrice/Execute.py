import pandas as pd

from Model import ModelFunctions
from ScrapeData import ScrapeDataFunctions
from ProcessData import ProcessDataFunctions

model_functions = ModelFunctions()
scrapedata_functions = ScrapeDataFunctions(2, 1, 2023)
processdata_functions = ProcessDataFunctions('./')

# Parameters
epochs = 5
window_size = 7
batch_size = 16
split_index = 110
learning_rate = 4e-4
shuffle_buffer_size = 1000

# Load the data
# _ = scrapedata_functions.scrape_data()
data = processdata_functions.clean_data()
scaled_data, store_scalers = processdata_functions.scale_data(data)

# Develop models for all fish types
fish_types = ['Ikan Bandeng', 'Ikan Tongkol', 'Ikan Kembung']
store_models = {}
for fish_type in fish_types:
    train_target, _, _, _ = processdata_functions.prepare_dataset(scaled_data, split_index, fish_type)
    model = model_functions.develop_model(train_target, window_size, batch_size, shuffle_buffer_size, learning_rate, epochs)
    store_models[fish_type] = model

# Create forecast
provinces = scaled_data['Provinsi'].unique()
fish_types = ['Ikan Bandeng', 'Ikan Tongkol', 'Ikan Kembung']
forecast_dict = {'Ikan Bandeng':[], 'Ikan Tongkol':[], 'Ikan Kembung':[], 'Provinsi':provinces}
for fish_type in fish_types:
    store_forecasts = []
    model = store_models[fish_type]
    for province in provinces:
        fish_price_province = scaled_data.loc[scaled_data['Provinsi'] == province, fish_type]
        window_fish_price_province = fish_price_province.values[-window_size:]
        scaler = store_scalers[fish_type][province]
        forecast = model_functions.model_forecast(model, window_fish_price_province, window_size, batch_size, scaler)
        store_forecasts.append(forecast[0][0])
    forecast_dict[fish_type] = store_forecasts
forecast_dataset = pd.DataFrame(forecast_dict)

print(forecast_dataset)