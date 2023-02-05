""" Define SAVEDataset class
"""

from dataclasses import dataclass
from base_dataset import BaseDataset
from utils import class_decorator, timeit

from collections import defaultdict
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from typing import Tuple

@class_decorator(timeit)
@dataclass
class SAVEDataset(BaseDataset):
    """Class of load and process save energy and infodata
    """

    def __init__(self, start_date: str = '2017-01-01 00:00:00'):
        self.timestamp_form = '%Y-%m-%d %H:%M'
        self.start_date = pd.to_datetime(start_date)
        _, self.energy = self.preprocess_energy()
        self.info = self.load_infodata(csv_path='data/SAVE/csv/save_household_survey_data/save_household_survey_data_v0-3.csv')
        self.energy, self.info = self.align_dataset(self.energy, self.info)
        
    def load_and_merge_energy_multiple_households(self, datadir: str) -> pd.DataFrame:
        """Merge multiple raw energy datasets

        Args:
            datadir (str): base dir to save data.

        Returns:
            pd.DataFrame: merged energy data
                          columns: bmg_id, received_timestamp, recorded_timestamp, energe
        """
        data_dir_list = ['save_consumption_data_2017_1_v0-1/',]
                        #  'save_consumption_data_2017_2_v0-1/',
                        #  'save_consumption_data_2018_1_v0-1/',
                        #  'save_consumption_data_2018_2_v0-1/']
        data_list = []
        for data_dir in data_dir_list:
            data_path = join(datadir, data_dir)
            file_list = [join(data_path, f) for f in listdir(
                data_path) if isfile(join(data_path, f))]
            for file in file_list:
                data = pd.read_csv(file, low_memory=False)
                data_list.append(data)
        energy_raw = pd.concat(data_list, axis=0)
        return energy_raw
    
    def get_inst_energy(self, cuml_energy: pd.DataFrame) -> pd.DataFrame:
        """Get instantaneous Energy consumption from the cumulative value

        Args:
            cuml_energy (pd.DataFrame): Energy consumption for the day indicated on the wattmeter

        Returns:
            pd.DataFrame: Energy consumption for 15 minutes (It is measured every 15 minutes)
        """
        inst_energy = cuml_energy.diff()
        # fill the initial value with the next of
        inst_energy.iloc[0] = inst_energy.iloc[1]
        return inst_energy
    
    def convert_to_timeseries(self, energy_raw: pd.DataFrame) -> pd.DataFrame:
        """convert raw data in unix timestamp to timeseries data with datetime manner

        Args:
            energy_raw (pd.DataFrame): raw energy data. one sample is a single energy 
                value of a single household for an instant times

        Returns:
            pd.DataFrame: timeseries data with datetime. 
                index is timestamp and column is each households
        """

        # convert unix timestamp to datetime
        energy_raw['timestamp'] = pd.to_datetime(energy_raw['recorded_timestamp'],unit='s')

        # Make empyty dataframe with timeseries manner
        start_time = energy_raw['timestamp'].min()
        end_time = energy_raw['timestamp'].max()
        date_range = pd.date_range(start=start_time, end=end_time, freq='15min', closed=None)
        date_range = pd.DataFrame(date_range, columns = ['timestamp']).set_index('timestamp')
        
        def rename_col(key, value):
            value = value.rename(columns = {'energy':key})[['timestamp',key]]
            return value.set_index('timestamp')
        
        # group the data by 'bmg_id' and convert to a list
        energy_grouped = energy_raw.groupby('bmg_id')

        # rename the columns in each group
        renamed_list = [rename_col(name, group) for name, group in energy_grouped]
        renamed_list = [date_range] + renamed_list

        # concatenate the dataframes along the columns axis
        energy = pd.concat(renamed_list, axis=1, join='outer')
        
        return energy
        
    def unify_units(self, energy: pd.DataFrame) -> pd.DataFrame:
        """get instaneous energy consumption with kWH from raw data

        Args:
            energy (pd.DataFrame): raw energy consumption

        Returns:
            pd.DataFrame: instaneous energy consumption with kWH
        """
        energy = self.get_inst_energy(energy)
        energy = self.wh_to_kwh(energy)
        energy = self.trunc_data(energy, self.start_date)
        return energy

    def aggregate_household_energy_data(self, energy: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate energy data from different households with similar codes.
        Household codes can vary in case (e.g. a3956661356, A3956661356).

        Args:
            energy (pd.DataFrame): time series energy data

        Returns:
            pd.DataFrame: consolidated time series energy data
        """

        household_codes = defaultdict(list)
        for col in energy.columns:
            household_code = col[1:].lower()
            household_codes[household_code].append(col)

        columns_to_drop = []
        for _, columns in household_codes.items():
            if len(columns) >= 2:
                first_column, second_column = columns[:2]
                non_null_indices = ~energy[second_column].isnull()
                energy.loc[non_null_indices, first_column] = energy.loc[non_null_indices, second_column]
                columns_to_drop.append(second_column)

        energy.drop(columns=columns_to_drop, inplace=True)
        energy.columns = [col[1:].lower() for col in energy.columns]

        return energy


    def load_infodata(self, csv_path: str) -> pd.DataFrame:
        """
        Load infodata csv to pandas dataframe, sort by 'InterviewDate' and transpose the dataframe.
        Also cast column names to string.
        
         Args:
            csv_path (str): infodata csv file path

        Returns:
            pd.DataFrame: infodata data in pandas dataframe format
        """

        info = pd.read_csv(csv_path, index_col=0)

        # Interviedate가 빠른 순으로 정렬
        info.sort_values('InterviewDate', inplace=True)
        info = info.T
        info.columns = info.columns.astype(str)
        return info

    def align_dataset(self, energy: pd.DataFrame, info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align the order of the labels in the infodata with the data in the energy dataset.
        
        Args:
            energy: Pandas DataFrame with the energy data
            info: Pandas DataFrame with the infodata information
        
        Returns:
            Tuple of two Pandas DataFrames: aligned energy data and infodata
        """
        # Initialize list to store valid columns in both dataframes
        valid_col = []
        
        # Loop through columns in the energy dataframe
        for col in energy.columns:
            # Check if the column is also in the infodata dataframe
            if col in info.columns:
                # If it is, append to the list of valid columns
                valid_col.append(col)
        
        # Select only the valid columns in the infodata dataframe
        info = info[valid_col].T
        # Select only the valid columns in the energy dataframe
        energy = energy[valid_col]
        
        # Return the aligned energy data and infodata
        return energy, info

    def replace_invalid_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """replace invalid energy consumption value which is equals to 0

        Args:
            df (pd.DataFrame): Energy consumption data

        Returns:
            pd.DataFrame: replaced energy consumption data
        """
        df[df == 0] = np.nan
        return df

    def preprocess_energy(self):
        energy_raw = self.load_and_merge_energy_multiple_households(datadir='data/SAVE/csv/')
        energy = self.convert_to_timeseries(energy_raw=energy_raw)
        energy = self.aggregate_household_energy_data(energy=energy)
        energy = self.replace_invalid_value(energy)
        energy = self.unify_units(energy)
        return energy_raw, energy


if __name__ == '__main__':
    
    # local test
    save = SAVEDataset()
    