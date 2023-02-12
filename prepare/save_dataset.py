""" Define SAVEDataset class
"""

from dataclasses import dataclass
from base_dataset import BaseDataset
from utils.logger import class_decorator, timeit

from collections import defaultdict
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

@class_decorator(timeit)
@dataclass
class SAVEDataset(BaseDataset):
    """Class of load and process save energy and infodata
    """

    def __init__(self, start_date: str = '2018-01-01 00:00:00', end_date = '2018-06-30 23:45:00'):
        self.timestamp_form = '%Y-%m-%d %H:%M'
        
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self.info = self.preprocess_info()
        _, self.energy = self.preprocess_energy()
        self.energy, self.info = self.align_dataset(self.energy, self.info)
        
        self.save_prepared_data(self.energy, self.info)
        
    def save_prepared_data(self, energy, info):
        energy.to_csv('data/prepared/SAVE/energy.csv')
        info.to_csv('data/prepared/SAVE/info.csv')
    
    def load_and_merge_energy_multiple_households(self, datadir: str) -> pd.DataFrame:
        """Merge multiple raw energy datasets

        Args:
            datadir (str): base dir to save data.

        Returns:
            pd.DataFrame: merged energy data
                          columns: bmg_id, received_timestamp, recorded_timestamp, energe
        """
        data_dir_list = ['save_consumption_data_2017_1_v0-1/',
                         'save_consumption_data_2017_2_v0-1/',
                         'save_consumption_data_2018_1_v0-1/',
                         'save_consumption_data_2018_2_v0-1/']
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
        energy = self.trunc_data(energy, self.start_date, self.end_date)
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
        energy.columns = [col[2:].lower() for col in energy.columns]

        return energy


    def load_surveydata(self, csv_path: str) -> pd.DataFrame:
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
        info = info.T
        return info

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
        """ Preprocess the raw energy data by loading, merging, converting to time series, aggregating, replacing invalid values and unifying the units.

        Returns:
            energy_raw (pd.DataFrame): Raw energy data loaded from multiple households.
            energy (pd.DataFrame): Processed energy data with unified units and replaced invalid values.
        """
        energy_raw = self.load_and_merge_energy_multiple_households(datadir='data/raw/SAVE/csv/')
        energy = self.convert_to_timeseries(energy_raw=energy_raw)
        energy = self.aggregate_household_energy_data(energy=energy)
        energy = self.replace_invalid_value(energy)
        energy = self.unify_units(energy)
        return energy_raw, energy
    
    def preprocess_info(self) -> pd.DataFrame:
        """
        This is a method for preprocessing household info data. The method performs the following operations on the data:

        Loads the information data from a CSV file and converts it into a Pandas dataframe.

        Extracts the following information from the data and stores it in a new Pandas dataframe "info":

        a. Number of residents (Q1)
        b. Single or not (Q2)
        c. Retired or not (Q3)
        d. Age of chief income earner (Q4)
        e. Age of building (Q5)
        f. House type (Q6)
        g. Number of bedrooms (Q7)

        Converts the extracted data into the desired format and stores the result in the "info" dataframe. For example, the age of the chief income earner is divided into three categories, and the number of bedrooms is quantized into three categories.

        Returns:
            pd.DataFrame: final preprocessed information.
        """
        survey = self.load_surveydata(csv_path='data/raw/SAVE/csv/save_household_survey_data/save_household_survey_data_v0-3.csv')
        info = pd.DataFrame(index = survey.index)

        ### Q1
        # Number of residents
        ###
        info['Q1'] = survey['Q2'].values

        ### Q2
        # Single
        ###
        info['Q2'] = survey['Q2'].values == 1

        ### Q3
        # retired or not
        ###
        info['Q3'] = survey['Q2D'].values # retired or not
        info['Q3'] = (info['Q3'] == 7)

        ### Q4 
        # Age of chief income earner
        # 1: Under 18
        # 2: 18 - 24
        # 3: 25 - 34
        # 4: 35 - 44
        # 5: 45 - 54
        # 6: 55 - 64
        # 7: 65 - 74
        # 8: 75+
        # 9: refused
        ###

        info['Q4'] = survey['Q2B'].values
        tmp_arr = np.zeros(info['Q4'].shape)
        tmp_arr[:] = np.nan
        tmp_arr[(info['Q4'] >= 7).values] = 2
        tmp_arr[(info['Q4'] <= 6).values] = 1
        tmp_arr[(info['Q4'] <= 3).values] = 0
        info['Q4'] = tmp_arr.copy()

        ### Q5 
        # Age of building
        # 기준연도: 2017
        # 1: Pre 1850
        # 2: 1850 to 1899
        # 3: 1900  to 1918
        # 4: 1919 to 1930
        # 5: 1931 to 1944
        # 6: 1945 to 1964
        # 7: 1965 to 1980
        # 8: 1981 to 1990
        # 9: 1991 to 1995
        # 10: 1996 to 2001
        # 11: 2002 or later
        # 12: Don't know
        # 13: Refused
        # 14: Not asked

        # 9,10,11 : new
        # 1 ~ 8: old
        ###
        info['Q5'] = np.nan
        row, col = np.where(survey.loc[:,'Q3_15_1':'Q3_15_14'] == 1)
        for r, c in zip(row, col):
            info.iloc[r, -1] = c + 1
        tmp_arr = np.zeros(info['Q5'].shape)
        tmp_arr[:] = np.nan
        tmp_arr[info['Q5'] >= 9] = 1
        tmp_arr[info['Q5'] < 9] = 0
        info['Q5'] = tmp_arr.copy()

        ### Q6
        # house type
        # 1 = Detached
        # 2 = Semi detached
        # 3 = Terraced or end terraced
        # 4 = In a purpose-built block of flats or tenement
        # 5 = Part of a converted or shared house (including bedsits)
        # 6 = In a commercial building (for example in an office building, hotel, or over a shop)
        # 7 = A caravan or other mobile or temporary structure
        # 8 = Refused
        ###
        info['Q6'] = survey['Q8_2'].values
        (info['Q6'] == 1).sum()
        (info['Q6'] == 2).sum()
        (info['Q6'] == 3).sum()
        tmp_arr = np.zeros(info['Q6'].shape)
        tmp_arr[:] = np.nan
        tmp_arr[info['Q6']  == 1] = 0
        tmp_arr[info['Q6'] == 2] = 1
        tmp_arr[info['Q6'] == 3] = 1

        info['Q6'] = tmp_arr.copy()

        ### Q7
        # Number of bedrooms
        # Numeric
        # 0~13개 -- quantize 필요
        ###

        info['Q7'] = survey['Q8_7'].values
        tmp_arr = np.zeros(info['Q7'].shape)
        tmp_arr[:] = np.nan
        tmp_arr[info['Q7'] == 1] = 0
        tmp_arr[info['Q7'] == 2] = 0
        tmp_arr[info['Q7'] ==3] = 1
        tmp_arr[info['Q7'] == 4] = 2
        tmp_arr[info['Q7'] > 4] = 3
        info['Q7'] = tmp_arr.copy()

        return info


if __name__ == '__main__':
    
    # local test
    save = SAVEDataset()
    