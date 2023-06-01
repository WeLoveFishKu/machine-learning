import glob 
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class ProcessDataFunctions:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_scraped_data(self):
        """Combines all raw scraped datas, then retrieve the necessary information from the data
        Returns:
        data (pandas dataframe) - dataframe that contains all the necessary data used for the model
        """
        csv_files = glob.glob(self.folder_path + "*.csv")
        data = pd.DataFrame()
        for file in csv_files:
            temp_data = pd.read_csv(file).set_index('Komoditas (Rp)').transpose()
            temp_data['Provinsi'] = [file[16:-4] for _ in range(len(temp_data))]
            data = pd.concat([data, temp_data])
        data = data.loc[:, ['Ikan Kembung', 'Ikan Tongkol', 'Ikan Bandeng', 'Provinsi']].reset_index().rename(columns={'index':'Date'})

        return data

    def clean_data(self, data):
        """Drop any missing values rows in the data and reformat the date column
        Args:
        data (pandas dataframe) - dataframe that contains all the necessary data used for the model
        
        Returns:
        data (pandas dataframe) - dataframe that are clear from both missing and wrong format date values
        """
        columns = ['Ikan Kembung', 'Ikan Tongkol', 'Ikan Bandeng']
        for col in columns:
            data[col] = data[col].apply(lambda row: np.nan if row == '-' else float(row))
        data = data.dropna().reset_index().drop(columns=['index'])
        data['Date'] = data['Date'].apply(lambda row: datetime.strptime(row, '%d/%m/%Y').date())

        return data

    def _calc_probability(self, data):
        """Calculates the probability of a value coming from a normal distribution
        Args:
        data (pandas dataframe) - localized dataframe

        Returns:
        probability (pandas dataframe) - dataframe that contains the probability of the value coming from a normal distribution
        """
        probability = data.copy()
        for col in data.columns[1:-1]:
            mu = data[col].mean()
            var = data[col].var()**0.5
            probability[col] = probability[col].apply(lambda x: norm.cdf((x - mu)/var))
            
        return probability

    def _find_anomalies(self, probability):
        """Check each value that has the probability lower than 0.15
        Args:
        probability (pandas dataframe) - dataframe that contains the probability of the value coming from a normal distribution

        Returns:
        check_anomaly (pandas dataframe) - dataframe that contains booleans on whether the probability of the value appearing in the normal distributionis lower than 0.15
        """
        check_anomaly = probability.copy()
        for col in check_anomaly.columns[1:-1]:
            check_anomaly[col] = check_anomaly[col].apply(lambda x: x < 0.15)

        return check_anomaly
    
    def _fix_local_anomalies(self, data, check_anomaly):
        """Fixes anomalies by substitute it with the mean price in the previous 7 days
        Args:
        data (pandas dataframe) - localized dataframe
        check_anomaly (pandas dataframe) - dataframe that contains booleans on whether the probability of the value appearing in the normal distributionis lower than 0.15

        Returns:
        fixed_data (pandas dataframe) - localized dataframe that is clear from any anomalies
        """
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

    def fix_global_anomalies(self, clean_data):
        """Fixes all anomalies within the dataframe
        Args:
        clean_data (pandas dataframe) - dataframe that are clear from both missing and wrong format date values

        Returns:
        fixed_data (pandas dataframe) - dataframe that is clear from any anomalies
        """
        fixed_data = pd.DataFrame()
        for province in clean_data['Provinsi'].unique():
            data_province = clean_data.loc[clean_data['Provinsi'] == province]
            probability = self._calc_probability(data_province)
            check_anomaly = self._find_anomalies(probability)
            fixed_data_province = self._fix_local_anomalies(data_province, check_anomaly)
            fixed_data = pd.concat([fixed_data, fixed_data_province])

        return fixed_data
    
    def scale_data(self, fixed_data):
        """Scales the dataframe into having the mean of 0 and variance of 1
        Args:
        fixed_data (pandas dataframe) - dataframe that is clear from any anomalies

        Returns:
        scaled_data (pandas dataframe) - dataframe that has been scaled into having the mean of 0 and variance of 1
        store_scalers (dict) - store the scaler used to revert the scaled data
        """
        provinces = fixed_data['Provinsi'].unique()
        fish_types = fixed_data.columns[1:-1]
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
        target_data (array of float) - contains the target data used for training
        feature_data (numpy array) - contains the features data
        split_index (int) - the number of data contained in the training data

        Returns:
        train_target (array of float)  - contains the first split_index target data
        train_feature (numpy array) - contains the first split_index target data
        test_target (array of float) - contain the final remainder of the target data that didn't make it into the train_target
        test_feature (numpy array) - contain the final remainder of the feature data that didn't make it into the train_feature
        """
        train_target = target_data[:split_index]
        train_feature = feature_data[:split_index]
        test_target = target_data[split_index:]
        test_feature = feature_data[split_index:]

        return (train_target, train_feature, test_target, test_feature)
    
    def prepare_dataset(self, scaled_data, split_index, fish_type):
        """Prepare the train and test dataset for both the target and feature data
        Args:
        scaled_data (pandas dataframe) - dataset that has been scaled into having the mean of 0 and variance of 1
        split_index (int) - the number of data contained in the training data
        fish_type (str) - the type of fish that will be aggregated from the scaled_data

        Returns:
        train_target (array of float) - contain the first split_index's amount of target data
        train_feature (numpy array) - contain the first split_index's amount of feature data
        test_target (array of float) - contain the final remainder of the target data that didn't make it into the train_target
        test_feature (numpy array) - contain the final remainder of the feature data that didn't make it into the train_feature
        """
        provinces = scaled_data['Provinsi'].unique()
        train_target, train_feature, test_target, test_feature = np.array([]), np.array([], dtype='datetime64'), np.array([]), np.array([], dtype='datetime64')
        for province in provinces:
            data_provinsi = scaled_data.loc[scaled_data['Provinsi'] == province, ['Date', fish_type]]
            price_data = data_provinsi[fish_type].values
            date_data = data_provinsi['Date'].values
            temp_train_target, temp_train_feature, temp_test_target, temp_test_feature = self.slice_dataset(price_data, date_data, split_index)
            train_target = np.concatenate((train_target, temp_train_target))
            train_feature = np.concatenate((train_feature, temp_train_feature))
            test_target = np.concatenate((test_target, temp_test_target))
            test_feature = np.concatenate((test_feature, temp_test_feature))
        
        return (train_target, train_feature, test_target, test_feature)