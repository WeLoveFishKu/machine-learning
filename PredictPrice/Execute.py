import time
import schedule
import pandas as pd
from datetime import date
from keras.models import load_model

from Model import ModelFunctions
from ScrapeData import ScrapeDataFunctions
from ProcessData import ProcessDataFunctions

model_functions = ModelFunctions()
processdata_functions = ProcessDataFunctions()
scrapedata_functions = ScrapeDataFunctions(1, 1, 2023)

def execute(t, model_bandeng=None, model_kembung=None, model_tongkol=None):
    # Load the data
    store_df = scrapedata_functions.scrape_data()
    data = processdata_functions.load_scraped_data(store_df)
    clean_data = processdata_functions.clean_data(data)
    fixed_data = processdata_functions.fix_global_anomalies(clean_data)
    scaled_data, store_scalers = processdata_functions.scale_data(fixed_data)

    # Parameters
    epochs = 1250
    window_size = 7
    batch_size = 16
    split_index = 115
    learning_rate = 4e-4
    shuffle_buffer_size = 1000

    if model_bandeng == None and model_kembung == None and model_tongkol == None:
        # Develop models for all fish types
        fish_types = ['Ikan Bandeng', 'Ikan Tongkol', 'Ikan Kembung']
        store_models = {}
        for fish_type in fish_types:
            train_target, _, _, _ = processdata_functions.prepare_dataset(scaled_data, split_index, fish_type)
            model = model_functions.develop_model(train_target, window_size, batch_size, shuffle_buffer_size, learning_rate, epochs)
            model.save(f'model_PredictPriceIkan{fish_type[5:]}.h5')
            store_models[fish_type] = model
    
    else:
        store_models = {'Ikan Bandeng': load_model(model_bandeng),
                        'Ikan Tongkol': load_model(model_kembung),
                        'Ikan Kembung': load_model(model_tongkol)}

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
    forecast_dataset.to_csv(f'FishPriceForecast_{date.today()}.csv', index=False)

    return forecast_dataset

schedule.every().day.at("10:38").do(execute, 'Program is currently running', 'model_PredictPriceIkanBandeng.h5', 
                                                                            'model_PredictPriceIkanKembung.h5', 
                                                                            'model_PredictPriceIkanTongkol.h5')

while True:
    schedule.run_pending()
    time.sleep(60)