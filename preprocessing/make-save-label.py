# %%
import pandas as pd

### 라벨 로드
SAVE_label = pd.read_csv('data/SAVE/save_household_survey_data_v0-3.csv', index_col = 0)
# Interviedate가 빠른 순으로 정렬
SAVE_label.sort_values('InterviewDate', inplace = True)
SAVE_label = SAVE_label.T
SAVE_label.columns = SAVE_label.columns.astype(str)
SAVE_label = SAVE_label.T

# %%
survey = pd.DataFrame()

# %% Q1
'''
Number of residents
'''
survey['Q1'] = SAVE_label['Q2'].values

# %% Q2
'''
Single
'''
survey['Q2'] = SAVE_label['Q2'].values

# %% Q3
'''
retired or not
'''
survey['Q3'] = SAVE_label['Q2D'].values # retired or not
survey['Q3'] = (survey['Q3'] == 7).astype(int)

# %% Q4 
'''
Age of chief income earner
1: Under 18
2: 18 - 24
3: 25 - 34
4: 34 - 44
5: 45 - 54
6: 55 - 64
7: 65 - 74
8: 75+
9: refused
'''
survey['Q4'] = SAVE_label['Q2B'].values

# %% Q5 
'''
Age of building
기준연도: 2017
1: Pre 1850
2: 1850 to 1899
3: 1900  to 1918
4: 1919 to 1930
5: 1931 to 1944
6: 1945 to 1964
7: 1965 to 1980
8: 1981 to 1990
9: 1991 to 1995
10: 1996 to 2001
11: 2002 or later
12: Don't know
13: Refused
14: Not asked
'''
survey['Q5'] = SAVE_label['Q3_15'].values

# %% Q6
'''
house type
1 = Detached
2 = Semi detached
3 = Terraced or end terraced
4 = In a purpose-built block of flats or tenement
5 = Part of a converted or shared house (including bedsits)
6 = In a commercial building (for example in an office building, hotel, or over a shop)
7 = A caravan or other mobile or temporary structure
8 = Refused
'''
survey['Q6'] = SAVE_label['Q8_2'].values

# %% Q7
'''
Number of bedrooms
Numeric
0~13개 -- quantize 필요
'''
survey['Q7'] = SAVE_label['Q8_7'].values
