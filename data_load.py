"""
Define energy dataset
"""

import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np

class HouseholdEnergyDataset(Dataset):
    """household energy dataset
    """
    def __init__(self, data_dir, label_option, sampling_rate = None):
        data_raw, label_raw = self.load_dataset(data_dir)
        label = self.get_label_from_option(label_raw, label_option)
        if sampling_rate:
            data_raw = self.downsampling(data_raw, sampling_rate=sampling_rate)
        data = self.transform(data_raw)
        data, label= self.remove_invalid_household(data, label)
        self.data = data
        self.label = label
        assert len(data) == len(label)
        
    def get_label_from_option(self, label_raw, label_option):
        """ 
        label column is Q1, Q2, ...
        get 1d array
        """
        label = label_raw[f'Q{label_option}'].values
        return label
        
    def remove_invalid_household(self, data, label):
        """ Remove invalid label with is nan
        """
        invalid_idx = pd.isnull(label)
        data = data[~invalid_idx]
        label = label[~invalid_idx]
        return data, label
        
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)
    
    def load_dataset(self, data_dir):
        data = pd.read_csv(os.path.join(data_dir, 'energy.csv'), index_col = 0)
        label = pd.read_csv(os.path.join(data_dir, 'info.csv'), index_col = 0)
        return data, label
    
    def downsampling(self, data_raw, sampling_rate=2, datatype='energy'):
        """ 
        Downsampling data.
        If the datatype is energy, return summed data
        Else if the datatype is load, return averaged data
        """
        n = data_raw.shape[0]
        list_ = []
        time_ = []
        for i in range(0, n, sampling_rate):
            data = data_raw.iloc[i:i+sampling_rate,:]
            invalid_data_idx = np.any(pd.isnull(data), axis=0)
            if datatype == 'energy':
                data = data.sum(axis=0)
            elif datatype == 'load':
                data = data.mean(axis=0)
            data.iloc[invalid_data_idx] = np.nan
            list_.append(data)
            time_.append(data_raw.index[i])
        downsampled_data = pd.concat(list_, axis=1).T
        downsampled_data.index = time_
        return downsampled_data
        
    def remove_invalid_sample(self, data_2d, home_index):
        """remove invalid sample such as all nan load or constant load
        """
        
        # handle invalid values
        invalid_idx = np.isnan(data_2d).any(axis=1)
        data_2d = data_2d[~invalid_idx, :]
        home_index = home_index[~invalid_idx]

        # constant load filtering
        constant_load = np.all(np.diff(data_2d, axis=1) == 0, axis=1)
        data_2d = data_2d[~constant_load, :]
        home_index = home_index[~constant_load]
        return data_2d, home_index
    
    def get_representative_loads(self, data_2d, home_index):
        """
        Given a 2D data array and the home index of each data point, this function calculates
        the representative loads for each home.
        
        Args:
            data_2d (np.ndarray): 2D array of data, where each row represents a data point.
            home_index (np.ndarray): 1D array of home indices, where each element corresponds to a data point in data_2d.
            
        Returns:
            np.ndarray: 2D array of representative loads, where each row represents the representative load for a home.
        """
        
        def calculate_distance_matrix(data):
            """ Calculates the pairwise Euclidean distances between all data points in the given data array.
            """
            n_samples = data.shape[0]
            distance = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    distance[i, j] = np.linalg.norm(data[i,:]-data[j,:])
            return distance

        unique_home_index = np.unique(home_index)
        representative_loads = []
        
        for home_idx in unique_home_index:
            # Select data points that belong to the current home
            data = data_2d[home_index == home_idx,:]
            
            # Calculate the mean pairwise distance between all data points in the current home
            distance = calculate_distance_matrix(data).mean(axis=0)
            
            # Calculate the mean and standard deviation of the mean pairwise distances
            mean_ = distance.mean()
            std_ = np.sqrt(distance.var())

            # Select data points that have mean pairwise distances below the mean - standard deviation
            idx = distance <= mean_ - std_
            
            # Ensure that there are at least 5 data points selected
            if len(idx) <= 5:
                idx[:] = True
            if idx.sum() <= 5:
                idx[:] = True

            # Select the data points that have mean pairwise distances below the mean - standard deviation
            data = data[idx,:]
            
            # Calculate the mean of the selected data points as the representative load for the current home
            representative_loads.append(data.mean(axis=0))

        data_rep = np.array(representative_loads)
        return data_rep

    def transform(self, data, sampling_interv = 24 * 2 * 7):
        """
        Make time series dataset in daily or weekly dataset depending on sampling_interv
        
        Returns:
            np.array: 2d array
            np.array: home index array
        """

        # dataframe => 3d numpy array
        n_d, n_h = data.shape
        n_w = n_d // sampling_interv
        n_d = n_w * sampling_interv
        df_rs = data.iloc[:n_d, :].values.T.reshape(n_h, -1, sampling_interv)

        # 3d numpy array => 2d numpy array
        n, m, l = df_rs.shape
        data_2d = df_rs.reshape(n * m, l)
        home_index = np.tile(np.arange(0, n), m)
        
        data_2d, home_index = self.remove_invalid_sample(data_2d, home_index)
        
        data_rep= self.get_representative_loads(data_2d, home_index)
        return data_rep
    
    
if __name__ == '__main__':
    save_dataset = HouseholdEnergyDataset(data_dir = 'data/prepared/SAVE', label_option = 1, sampling_rate = 2)
    cer_dataset = HouseholdEnergyDataset(data_dir = 'data/prepared/CER', label_option = 1)