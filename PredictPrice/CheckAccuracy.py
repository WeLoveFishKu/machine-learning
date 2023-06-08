import numpy as np
import pandas as pd

def calc_accuracy(real_values, pred_values):
    min_len = np.min([len(real_values), len(pred_values)])
    real_values = real_values[:min_len]
    pred_values = pred_values[:min_len]
    accuracies = np.abs(pred_values - real_values) / pred_values
    accuracy = np.mean(accuracies)

    return accuracy

real_data = pd.read_csv('FishPrice_2023-06-08.csv')
pred_data = pd.read_csv('FishPriceForecast_2023-06-07.csv')

latest_real_data = real_data.loc[real_data['Date'] == '2023-06-08']


bandeng_acc = calc_accuracy(latest_real_data['Ikan Bandeng'].values, pred_data['Ikan Bandeng'].values)
kembung_acc = calc_accuracy(latest_real_data['Ikan Kembung'].values, pred_data['Ikan Kembung'].values)
tongkol_acc = calc_accuracy(latest_real_data['Ikan Tongkol'].values, pred_data['Ikan Tongkol'].values)
mean_acc = np.mean([bandeng_acc, kembung_acc, tongkol_acc])

print(bandeng_acc)
print(kembung_acc)
print(tongkol_acc)
print(mean_acc)