import glob 
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class ProcessDataFunctions:
    def __init__(self):
        pass

    def load_scraped_data(self, folder_path):
        """Combines all raw scraped data and retrieve the necessary information from the data
        Args:
        folder_path - the path to the folder that contains all raw scraped data

        Returns:
        data - contains all the necessary data used for the model
        """
        csv_files = glob.glob(folder_path + "/*.csv")
        data = pd.read_csv(csv_files[0]).drop(columns=['Unnamed: 0']).set_index('Komoditas (Rp)').transpose()
        data['Provinsi'] = [csv_files[0][28:-4] for _ in range(len(data))]
        for file in csv_files[1:]:
            data_temp = pd.read_csv(file).drop(columns=['Unnamed: 0']).set_index('Komoditas (Rp)').transpose()
            data_temp['Provinsi'] = [file[28:-4] for _ in range(len(data_temp))]
            data = pd.concat([data, data_temp])
        data = data.loc[:, ['Ikan Kembung', 'Ikan Tongkol', 'Ikan Bandeng', 'Provinsi']]
        data = data.reset_index().rename(columns={'index':'Date'})

        return data

    def clean_data(self):
        """Drop rows from the data that contains any missing values, reformat the date column
        Returns:
        data - dataset that clear from any missing values and reformated date column
        """
        data = self.load_scraped_data()
        columns = ['Ikan Kembung', 'Ikan Tongkol', 'Ikan Bandeng']
        for col in columns:
            data[col] = data[col].apply(lambda row: np.nan if row == '-' else float(row))
        data = data.dropna().reset_index().drop(columns=['index'])
        data['Date'] = data['Date'].apply(lambda row: datetime.strptime(row, '%d/%m/%Y').date())

        return data

    def calc_probability(self, data):
        """Calculates the probabilites that data is in the normal distribution
        Args:
        data - dataset containing all informations for a province

        Returns:
        probability - dataset containing the probability of the data in the normal distribution
        """
        probability = data.copy()
        for col in data.columns[1:-1]:
            mu = data[col].mean()
            var = data[col].var()**0.5
            probability[col] = probability[col].apply(lambda x: norm.cdf((x - mu)/var))
            
        return probability

    def find_anomalies(self, data):
        """Checks each data entry that has the probability lower than 0.15
        Args:
        data - dataset containing all informations for a province

        Returns:
        check_anomaly - dataset containing booleans
        """
        check_anomaly = self.calc_probability(data)
        for col in check_anomaly.columns[1:-1]:
            check_anomaly[col] = check_anomaly[col].apply(lambda x: x < 0.15)

        return check_anomaly
    
    def fix_anomalies(self, data):
        """Fixes anomalies by find the mean price in the previous 7 days
        Args:
        data - dataset containing all informations for a province

        Returns:
        fixed_data - dataset containing all information for a province that has been clear from any anomalies
        """
        check_anomaly = self.find_anomalies(data)
        fixed_data = data.copy()
        for col in check_anomaly.columns[1:-1]:
            fixed_data_vals = fixed_data[col].values
            check_anomaly_vals = check_anomaly[col].values
            for i, val in enumerate(check_anomaly_vals):
                if val == True:
                    if i == 0:
                        mean = fixed_data_vals[i+1]
                    elif i - 7 < 0:
                        mean = np.mean(fixed_data_vals[:i])
                    else:
                        mean = np.mean(fixed_data_vals[i-7:i])
                    fixed_data_vals[i] = mean
            fixed_data[col] = fixed_data_vals
            
        return fixed_data

    def fix_all_provinces_anomalies(self, data):
        """Fix all data anomalies present in all provinces
        Args:
        data - dataset that clear from any missing values and reformated date column

        Returns:
        fixed_data - dataset that has been clear from any anomalies
        """
        fixed_data = pd.DataFrame({'Date':[], 'Ikan Kembung':[], 'Ikan Tongkol':[], 'Ikan Bandeng': [], 'Provinsi':[]})
        for province in data['Provinsi'].unique():
            data_province = data.loc[data['Provinsi'] == province]
            fixed_data_province = self.fix_anomalies(data_province)
            fixed_data = pd.concat([fixed_data, fixed_data_province])

        return fixed_data
    
    def scale_data(self, data):
        """Scales the data into having the mean of 0 and variance of 1
        Args:
        data - dataset that has been clear from any anomalies

        Returns:
        scaled_data - dataset that has been scaled into having the mean of 0 and variance of 1
        store_scalers - contains the scaler used to revert the scaled data
        """
        fixed_data = self.fix_all_provinces_anomalies(data)
        provinces = data['Provinsi'].unique()
        fish_types = data.columns[1:-1]
        store_scalers = {fish_type:{province:StandardScaler() for province in provinces} for fish_type in fish_types}
        temp_dict = {'Date':fixed_data['Date'].values}

        for fish_type in fish_types:
            temp_list = np.array([])
            scalers = store_scalers[fish_type]
            for province, scaler in scalers.items():
                fish_price = fixed_data.loc[fixed_data['Provinsi'] == province, fish_type].values.reshape(-1, 1)
                scaler.fit(fish_price)
                fish_price_scaled = scaler.transform(fish_price).T[0]
                temp_list = np.concatenate([temp_list, fish_price_scaled])
            temp_dict[fish_type] = temp_list
        
        temp_dict['Provinsi'] = fixed_data['Provinsi'].values
        scaled_data = pd.DataFrame(temp_dict)

        return (scaled_data, store_scalers)
    
    def slice_dataset(self, target_data, feature_data, split_index):
        """Slices the target and feature data to create training and testing data for both the target and feature data
        Args:
        target_data - contains the target data used for training
        feature_data - contains the features data
        split_index - the number of data contained in the training data

        Returns:
        test_feature - contains the first split_index target data
        train_feature - contains the first split_index target data
        test_target - contains the final split_index target data
        test_feature - contains the final split_index target data
        """
        # Get the train set 
        train_target = target_data[:split_index]
        train_feature = feature_data[:split_index]
        # Get the validation set
        test_target = target_data[split_index:]
        test_feature = feature_data[split_index:]

        return (train_target, train_feature, test_target, test_feature)
    
    def prepare_dataset(self, scaled_data, split_index, fish_type):
        provinces = scaled_data['Provinsi'].unique()
        train_price, train_date, test_price, test_date = np.array([]), np.array([], dtype='datetime64'), np.array([]), np.array([], dtype='datetime64')
        for province in provinces:
            data_provinsi = scaled_data.loc[scaled_data['Provinsi'] == province, ['Date', fish_type]]
            price_data = data_provinsi[fish_type].values
            date_data = data_provinsi['Date'].values
            temp_train_price, temp_train_date, temp_test_price, temp_test_date = self.slice_dataset(price_data, date_data, split_index)
            train_price = np.concatenate((train_price, temp_train_price))
            train_date = np.concatenate((train_date, temp_train_date))
            test_price = np.concatenate((test_price, temp_test_price))
            test_date = np.concatenate((test_date, temp_test_date))
        
        return (train_price, train_date, test_price, test_date)