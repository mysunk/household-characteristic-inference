""" Define CERDataset class
"""

from dataclasses import dataclass
from base_dataset import BaseDataset
from utils import class_decorator, timeit

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import copy
import os
from datetime import datetime, timedelta
from tqdm import tqdm

@class_decorator(timeit)
@dataclass
class CERDataset(BaseDataset):
    """Class of load and process cer energy and infodata
    """

    def __init__(self, start_date: str = '2010-01-01 00:00:00', end_date = '2010-06-30 23:30:00'):
        self.timestamp_form = '%Y-%m-%d %H:%M'
        
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.txt_to_csv()
        _, self.energy = self.preprocess_energy()
        
        self.info = self.preprocess_info()
        
        self.energy, self.info = self.align_dataset(self.energy, self.info)
        self.energy, self.info = self.filter_invalid_household(self.energy, self.info)
    
    def filter_invalid_household(self, energy, info):
        # filter all nan data
        nan_ratio = pd.isnull(energy).sum(axis=0) / energy.shape[0]
        invalid_idx = (nan_ratio == 1)
        energy = energy.loc[:,~invalid_idx]
        info = info.loc[~invalid_idx,:]
        return energy, info
        
    def preprocess_energy(self):
        energy_raw = self.load_and_merge_energy_multiple_households()
        energy = self.replace_invalid_value(energy_raw)
        return energy_raw, energy
        
    def load_and_merge_energy_multiple_households(self, datadir: str = 'data/raw/CER/files_csv/') -> pd.DataFrame:
        """Merge multiple raw energy datasets

        Args:
            datadir (str): base dir to save data.

        Returns:
            pd.DataFrame: merged energy data
                index is datetime and columns are household code
        """
        file_list = [f for f in listdir(datadir) if isfile(join(datadir, f))]
        home_list = []
        for i, file in tqdm(enumerate(file_list[:10])):
            home = pd.read_csv(datadir + file, low_memory=False, names=['time',file[:4]], skiprows=1)
            home.index = pd.to_datetime(home['time'])
            home.drop(columns=['time'],inplace=True)
            home_list.append(home)

        energy = pd.concat(home_list, axis=1)
        energy = self.trunc_data(energy, self.start_date, self.end_date)
        energy = energy.iloc[:2 * 24 * 365, :]  # 1year
        return energy
            
    def txt_to_csv(self, datadir = 'data/raw/CER/'): #TODO: make it faster is needed
        """
        Separate CER dataset File 1 ~6 . txt into .csv for each smart meter.
        """
        
        if os.exist(datadir + 'files_csv'):
            # already processed
            return
        
        def fillgap_interp(dataframe):
            """This function processes a given pandas dataframe and returns a pandas series with interpolated values.

            Args:
                dataframe (pandas.DataFrame): A dataframe with 2 columns, the first column being a time index and the second column the corresponding values.

            Returns:
                pandas.Series: A series with the same index as the processed dataframe but with the missing values interpolated.
            """
            ar = np.array(dataframe)
            startdate = datetime(2009, 1, 1) + timedelta(days=ar[0, 0] // 100 - 1)
            enddate = datetime(2009, 1, 1) + timedelta(days=ar[-1, 0] // 100 - 1) - timedelta(minutes=30)

            t = pd.date_range(startdate, enddate + timedelta(days=1), freq="30T")
            t = pd.to_datetime(t)

            duration = (enddate - startdate).days + 2
            val_ar = np.ones([duration, 48]) * -1
            for i in range(duration):
                for ii in range(48):
                    idx = (ar[0, 0] // 100 + i) * 100 + ii + 1
                    if ar[np.where(ar[:, 0] == idx), 1].size == 1:
                        val_ar[i, ii] = ar[np.where(ar[:, 0] == idx), 1]
                    elif ar[np.where(ar[:, 0] == idx), 1].size > 1:
                        val_ar[i, ii] = np.mean(ar[np.where(ar[:, 0] == idx), 1])
            print((val_ar == -1).sum())
            val_vec = val_ar.reshape([val_ar.size, 1])
            val_vec[np.where(val_vec == -1)] = float('nan')
            se_out = pd.Series(val_vec[:, 0], index=t)
            se_out.interpolate()

            return se_out
        
        def over48(dataframe):
            """This function takes a pandas dataframe as input and returns a new dataframe with only the rows whose index column is less than or equal to 48.
            
            Args:
                dataframe (pandas.DataFrame): A dataframe with 2 columns, the first column being a time index and the second column the corresponding values.

            Returns:
                pandas.DataFrame: A new dataframe with only the rows whose index column is less than or equal to 48.
            """
            ar = np.array(dataframe)
            ar = np.delete(ar, np.where(ar[:, 0] % 100 > 48), 0)

            df = pd.DataFrame(ar, columns=['index', 'usage'])
            return df
        
        for file_num in range(1, 7):
            filename = 'File' + str(file_num) + '.txt'

            df = pd.read_csv(datadir +'files/' + filename, sep=" ", names=["meter_id", "index", "usage"])

            meter_names = df['meter_id'].unique()

            cnt_normal = 0
            cnt_abnorm = 0
            for meters in meter_names:
                print(meters)
                temp_df = copy.deepcopy(df[df['meter_id'] == meters])
                temp_df.pop('meter_id')
                start_day = temp_df['index'].min() // 100
                end_day = temp_df['index'].max() // 100
                duration = end_day - start_day + 1
                temp_df = temp_df.sort_values(by=['index'])

                temp_df = over48(temp_df)
                print(pd.isnull(temp_df).sum())
                temp_df = fillgap_interp(temp_df)

                if temp_df.shape[0] / 48 == duration:
                    temp_df.to_csv(datadir +'files_csv/' + str(meters) + '_' + str(start_day) + '_' +
                                str(duration) + '_' + str(end_day) + '.csv',
                                mode='w')
                    cnt_normal = cnt_normal + 1
                else:
                    temp_df.to_csv(datadir +'files_csv/' + 'c' + str(meters) + '_' + str(start_day) + '_' +
                                str(duration) + '_' + str(end_day) + '(' + str(temp_df.shape[0] // 48) + ').csv',
                                mode='w')
                    cnt_abnorm = cnt_abnorm + 1

            print(str(cnt_normal) + ' normal and ' + str(cnt_abnorm) + ' abnormal are separated')
            print('Total ' + str(cnt_normal + cnt_abnorm))
    
    def load_surveydata(self, csv_path: str) -> pd.DataFrame:
        """
        Load infodata csv to pandas dataframe, sort by 'InterviewDate' and transpose the dataframe.
        Also cast column names to string.
        
         Args:
            csv_path (str): infodata csv file path

        Returns:
            pd.DataFrame: infodata data in pandas dataframe format
        """
        
        info = pd.read_csv(csv_path, encoding='cp1252', low_memory=False)
        info.set_index('ID', inplace=True)
        # replace blanks
        for j in range(info.shape[1]):
            for i in range(info.shape[0]):
                try:
                    int(info.iloc[i, j])
                except ValueError:
                    info.iloc[i, j] = -1

        for j in range(info.shape[1]):
            info.iloc[:, j] = info.iloc[:, j].astype(int)

        return info
    
    def preprocess_info(self):
        info = self.load_surveydata(csv_path = 'data/raw/CER/CER_Electricity_Data/Smart meters Residential pre-trial survey data.csv')
        
        # make empty dataframe
        survey = pd.DataFrame(index = info.index)

        ### Questions 1
        # Young (0): age <= 35
        # Med (1): 35 < age <= 65
        # High (2): 65 < age
        ###
        Q300 = 'Question 300: May I ask what age you were on your last birthday? INT: IF NECCESSARY, PROMPT WITH AGE BANDS'
        survey['Q1'] = -1

        idx = info[Q300] == 6
        survey.loc[idx, 'Q1'] = 2
        idx = (info[Q300] < 6).values * (info[Q300] > 2).values
        survey.loc[idx, 'Q1'] = 1
        idx = (info[Q300] <= 2).values
        survey.loc[idx, 'Q1'] = 0

        ### Questions 2
        # employed: 1
        # retired: 0
        ###
        Q310 = 'Question 310: What is the employment status of the chief income earner in your household, is he/she'
        survey['Q2'] = 1

        idx = (info[Q310] == 4).values + \
                (info[Q310] == 5).values + \
                (info[Q310] == 6).values + \
                (info[Q310] == 7).values
        survey.loc[idx, 'Q2'] = 0

        ### Questions 13, 3, 8, 9
        # 13
        # Few (1): #residents <=2
        # Many (2): #residents >2
        # 3
        # Have children: 1
        # not: 0
        # 8
        # Single: adults = 1 and children = 0
        #
        # 9
        # Family: adults > 1 and children > 0
        ###

        Q410 = 'Question 410: What best describes the people you live with? READ OUT'
        Q420 = 'Question 420: How many people over 15 years of age live in your home?'
        Q43111 = 'Question 43111: How many people under 15 years of age live in your home?'


        # 세대원 수가 1명인 경우
        survey['Q13'] = -1
        survey['Q3'] = -1
        survey.loc[info[Q410] == 1, 'Q13'] = 1
        survey.loc[info[Q410] == 1, 'Q3'] = 0

        # 모든 세대원이 15살 이상인 경우
        survey.loc[info[Q410] == 2, 'Q13'] = info.loc[info[Q410] == 2, Q420]
        survey.loc[info[Q410] == 2, 'Q3'] = 0

        # children이 있는 경우
        survey.loc[info[Q410] == 3, 'Q13'] = info.loc[info[Q410] == 3, Q420] + info.loc[info[Q410] == 3, Q43111]
        survey.loc[info[Q410] == 3, 'Q3'] = info.loc[info[Q410] == 3, Q43111]

        # children을 binary로
        survey.loc[survey['Q3'] >= 1, 'Q3'] = 1

        # is single?
        survey['Q8'] = 0
        idx = (survey['Q3'] == 0).values * (survey['Q13'] == 1).values
        survey.loc[idx, 'Q8'] = 1

        # is family?
        survey['Q9'] = 0
        num_adults = info[Q420] # except single
        idx = (survey['Q3'] > 0).values * (num_adults > 1).values
        survey.loc[idx, 'Q9'] = 1

        # #residents를 2 class로
        # survey.loc[survey['Q13'] <= 2, 'Q13'] = 0
        # survey.loc[survey['Q13'] > 2, 'Q13'] = 1

        ### Question 4
        # age of building
        # new (<=30): 0
        # old (>30) : 1
        ###
        Q4531 = 'Question 4531: Approximately how old is your home?'
        Q453 = 'Question 453: What year was your house built INT ENTER FOR EXAMPLE: 1981- CAPTURE THE FOUR DIGITS'
        survey['Q4'] = -1
        idx = (info[Q453] >= 1000) * (info[Q453] <= 2010)
        survey.loc[idx, 'Q4'] = 2009 - info.loc[idx, Q453]
        # 4, 5는 label 2 // 1,2,3은 label 1
        idx = (info[Q4531] != -1)
        old_idx = (info.loc[idx, Q4531] >= 4).index[(info.loc[idx, Q4531] >= 4).values]
        new_idx = (info.loc[idx, Q4531] <= 3).index[(info.loc[idx, Q4531] <= 3).values]
        survey.loc[old_idx, 'Q4'] = 50
        survey.loc[new_idx, 'Q4'] = 20

        idx = survey['Q4'] >= 30
        bad_idx = survey['Q4'] == -1
        survey.loc[idx,'Q4'] = 1
        survey.loc[~idx,'Q4'] = 0
        survey.loc[bad_idx,'Q4'] = -1

        ### Question 5
        # floar area
        # small: 0
        # med: 1
        # big: 2
        ###

        Q6103 = 'Question 6103: What is the approximate floor area of your home?'
        Q61031 = 'Question 61031: Is that'

        survey['Q5'] = info[Q6103].copy()

        bad_idx = info[Q6103] == 999999999

        idx = info[Q61031] == 2
        survey.loc[idx, 'Q5'] = survey.loc[idx, 'Q5'].values.copy() / 10.764

        idx = (survey['Q5'] <= 100).values
        idx3 = (survey['Q5'] > 200).values * (survey['Q5'] < 999999999).values
        idx2 = (survey['Q5'] > 100).values * (survey['Q5'] <= 200).values

        survey.loc[idx,'Q5'] = 0
        survey.loc[idx2,'Q5'] = 1
        survey.loc[idx3,'Q5'] = 2
        survey.loc[bad_idx,'Q5'] = -1
        survey['Q5'] = survey['Q5'].astype(int)

        np.unique(survey['Q5'], return_counts=True)

        #%% Question 6
        ###
        # Energy-efficient light bulb proportion
        # 0: up to half
        # 1: about 3 quarters of more
        ###
        Q4905 = 'Question 4905: And now considering energy reduction in your home please indicate the approximate proportion of light bulbs which are energy saving (or CFL)?  INT:READ OUT'

        # idx = info[Q4905] <= 3
        idx = info[Q4905] <= 2
        survey['Q6'] = 0
        survey.loc[idx, 'Q6'] = 0
        # idx = info[Q4905] >= 4
        idx = info[Q4905] >= 3
        survey.loc[idx, 'Q6'] = 1

        #%% Question 7
        # House type
        # Free : 0
        # Connected: 1
        ###

        Q4905 = 'Question 450: I would now like to ask some questions about your home.  Which best describes your home?'
        survey['Q7'] = -1

        survey.loc[info[Q4905] == 3, 'Q7'] = 0
        survey.loc[info[Q4905] == 5, 'Q7'] = 0

        survey.loc[info[Q4905] == 2, 'Q7'] = 1
        survey.loc[info[Q4905] == 4, 'Q7'] = 1

        # others
        survey.loc[(info[Q4905] <2).values * (info[Q4905] >5).values,'Q7'] = -1

        ### Question 10
        ###
        # Type of cooking facility
        # Electric: 1
        # non-electric: 0
        ###
        Q4704 = 'Question 4704: Which of the following best describes how you cook in your home'
        survey['Q10'] = 0
        idx = info[Q4704] == 1
        survey.loc[idx, 'Q10'] = 1

        ### Question 11
        # Number of bedrooms
        # very low (0): <=2
        # low (1): ==3
        # high (2): ==4
        # very high(3): > 4
        ###
        Q460 = 'Question 460: How many bedrooms are there in your home'
        survey['Q11'] = 0

        # very high
        idx = info[Q460] > 4
        survey.loc[idx, 'Q11'] = 3

        # high
        idx = info[Q460] == 4
        survey.loc[idx, 'Q11'] = 2

        # low
        idx = info[Q460] == 3
        survey.loc[idx, 'Q11'] = 1

        # very low
        idx = info[Q460] <= 2
        survey.loc[idx, 'Q11'] = 0

        bad_idx = info[Q460] == 6
        survey.loc[bad_idx, 'Q11'] = -1

        ### Question 12
        ###
        # Number of appliances
        # low (<=8)
        # med (8< <=11)
        # high (>11)
        ###
        Q49002 = ['Question 49002: Washing machine',
        'Question 49002: Tumble dryer',
        'Question 49002: Dishwasher',
        'Question 49002: Electric shower (instant)',
        'Question 49002: Electric shower (electric pumped from hot tank)',
        'Question 49002: Electric cooker',
        'Question 49002: Electric heater (plug-in convector heaters)',
        'Question 49002: Stand alone freezer',
        'Question 49002: A water pump or electric well pump or pressurised water system',
        'Question 49002: Immersion']
        Q490002=[
        'Question 490002: TV’s less than 21 inch',
        'Question 490002: TV’s greater than 21 inch',
        'Question 490002: Desk-top computers',
        'Question 490002: Lap-top computers',
        'Question 490002: Games consoles, such as xbox, playstation or Wii']

        survey['Q12'] = 0
        for question in Q49002:
            val = info.loc[:,question].values.copy()
            val[val<=1] = 0
            val[val == 2] = 1
            val[val == 3] = 2
            val[val == 4] = 3
            survey['Q12'] += val

        for question in Q490002:
            val = info.loc[:,question].values.copy()
            val[val<=1] = 0
            val[val == 2] = 1
            val[val == 3] = 2
            val[val == 4] = 3
            val[val == 5] = 4
            survey['Q12'] += val

        idx = survey['Q12'] <= 8
        idx2 = (survey['Q12'] > 8) * (survey['Q12'] <= 11)
        idx3 = survey['Q12'] > 11

        # survey.loc[idx, 'Q12'] = 0
        # survey.loc[idx2, 'Q12'] = 1
        # survey.loc[idx3, 'Q12'] = 2

        ### Questions 14
        # 0: AB
        # 1: C1 C2
        # 2: DE
        ###
        Q401 = 'Question 401: SOCIAL CLASS Interviewer, Respondent said that occupation of chief income earner was.... <CLASS> Please code'
        survey['Q14'] = -1
        idx = info[Q401] == 1
        survey.loc[idx, 'Q14'] = 0

        idx = (info[Q401] == 2).values + (info[Q401] == 3).values
        survey.loc[idx, 'Q14'] = 1

        idx = (info[Q401] == 4).values
        survey.loc[idx, 'Q14'] = 2

        ### Questions 15
        ###
        # 6시간 이상 unoccupied: 1
        # others: 0
        ###
        Q430 = 'Question 430: And how many of these are typically in the house during the day (for example for 5-6 hours during the day)'
        Q4312 = 'Question 4312: And how many of these are typically in the house during the day (for exanmple for 5-6 hours during the day)'

        survey['Q15'] = 0
        idx_1 = info[Q430] == 8
        idx_2 = info[Q4312] == 8
        survey.loc[idx_1, 'Q15'] = 1
        survey.loc[idx_2, 'Q15'] = 1

        # sort int Q
        sorted_col = ['Q'+str(i) for i in range(1,16)]
        survey = survey.reindex(sorted_col, axis=1)
        
        survey.index = survey.index.astype(str)

        return survey
    
if __name__ == '__main__':
    
    # local test
    cer = CERDataset()
    