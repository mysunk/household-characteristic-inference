# %%
import pandas as pd
import numpy as np

path = '../data/CER/'
survey_raw = pd.read_csv(path + 'Smart meters Residential pre-trial survey data.csv', encoding='cp1252', low_memory=False)
survey_raw.set_index('ID', inplace=True)
# replace blanks
for j in range(survey_raw.shape[1]):
    for i in range(survey_raw.shape[0]):
        try:
            int(survey_raw.iloc[i, j])
        except:
            survey_raw.iloc[i, j] = -1

for j in range(survey_raw.shape[1]):
    survey_raw.iloc[:, j] = survey_raw.iloc[:, j].astype(int)

# processed survey data
survey = pd.DataFrame(index = survey_raw.index)

#%% Queistion 확인
Qnum = '4312'
idx = np.where(np.array([Qnum in col for col in survey_raw.columns]))[0]
print(survey_raw.columns[idx])

#%% Questions 1
"""
Young (0): age <= 35
Med (1): 35 < age <= 65
High (2): 65 < age
"""
Q300 = 'Question 300: May I ask what age you were on your last birthday? INT: IF NECCESSARY, PROMPT WITH AGE BANDS'
survey['Q1'] = -1

idx = survey_raw[Q300] == 6
survey.loc[idx, 'Q1'] = 2
idx = (survey_raw[Q300] < 6).values * (survey_raw[Q300] > 2).values
survey.loc[idx, 'Q1'] = 1
idx = (survey_raw[Q300] <= 2).values
survey.loc[idx, 'Q1'] = 0

#%% Questions 2
"""
employed: 1
retired: 0
"""
Q310 = 'Question 310: What is the employment status of the chief income earner in your household, is he/she'
survey['Q2'] = 1

idx = (survey_raw[Q310] == 4).values + \
        (survey_raw[Q310] == 5).values + \
        (survey_raw[Q310] == 6).values + \
        (survey_raw[Q310] == 7).values
survey.loc[idx, 'Q2'] = 0

#%% Questions 13, 3, 8, 9
""" 13
Few (1): #residents <=2
Many (2): #residents >2
"""
""" 3
Have children: 1
not: 0
"""
""" 8
Single: adults = 1 and children = 0
"""
""" 9
Family: adults > 1 and children > 0
"""

Q410 = 'Question 410: What best describes the people you live with? READ OUT'
Q420 = 'Question 420: How many people over 15 years of age live in your home?'
Q43111 = 'Question 43111: How many people under 15 years of age live in your home?'


# 세대원 수가 1명인 경우
survey['Q13'] = -1
survey['Q3'] = -1
survey.loc[survey_raw[Q410] == 1, 'Q13'] = 1
survey.loc[survey_raw[Q410] == 1, 'Q3'] = 0

# 모든 세대원이 15살 이상인 경우
survey.loc[survey_raw[Q410] == 2, 'Q13'] = survey_raw.loc[survey_raw[Q410] == 2, Q420]
survey.loc[survey_raw[Q410] == 2, 'Q3'] = 0

# children이 있는 경우
survey.loc[survey_raw[Q410] == 3, 'Q13'] = survey_raw.loc[survey_raw[Q410] == 3, Q420] + survey_raw.loc[survey_raw[Q410] == 3, Q43111]
survey.loc[survey_raw[Q410] == 3, 'Q3'] = survey_raw.loc[survey_raw[Q410] == 3, Q43111]

# children을 binary로
survey.loc[survey['Q3'] >= 1, 'Q3'] = 1

# is single?
survey['Q8'] = 0
idx = (survey['Q3'] == 0).values * (survey['Q13'] == 1).values
survey.loc[idx, 'Q8'] = 1

# is family?
survey['Q9'] = 0
num_adults = survey_raw[Q420] # except single
idx = (survey['Q3'] > 0).values * (num_adults > 1).values
survey.loc[idx, 'Q9'] = 1

# #residents를 2 class로
# survey.loc[survey['Q13'] <= 2, 'Q13'] = 0
# survey.loc[survey['Q13'] > 2, 'Q13'] = 1

#%% Question 4
"""
age of building
new (<=30): 0
old (>30) : 1
"""
Q4531 = 'Question 4531: Approximately how old is your home?'
Q453 = 'Question 453: What year was your house built INT ENTER FOR EXAMPLE: 1981- CAPTURE THE FOUR DIGITS'
survey['Q4'] = -1
idx = (survey_raw[Q453] >= 1000) * (survey_raw[Q453] <= 2010)
survey.loc[idx, 'Q4'] = 2009 - survey_raw.loc[idx, Q453]
# 4, 5는 label 2 // 1,2,3은 label 1
idx = (survey_raw[Q4531] != -1)
old_idx = (survey_raw.loc[idx, Q4531] >= 4).index[(survey_raw.loc[idx, Q4531] >= 4).values]
new_idx = (survey_raw.loc[idx, Q4531] <= 3).index[(survey_raw.loc[idx, Q4531] <= 3).values]
survey.loc[old_idx, 'Q4'] = 50
survey.loc[new_idx, 'Q4'] = 20

idx = survey['Q4'] >= 30
bad_idx = survey['Q4'] == -1
survey.loc[idx,'Q4'] = 1
survey.loc[~idx,'Q4'] = 0
survey.loc[bad_idx,'Q4'] = -1

#%% Question 5
"""
floar area
small: 0
med: 1
big: 2
"""

Q6103 = 'Question 6103: What is the approximate floor area of your home?'
Q61031 = 'Question 61031: Is that'

survey['Q5'] = survey_raw[Q6103].copy()

bad_idx = survey_raw[Q6103] == 999999999

idx = survey_raw[Q61031] == 2
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
"""
Energy-efficient light bulb proportion
0: up to half
1: about 3 quarters of more
"""
Q4905 = 'Question 4905: And now considering energy reduction in your home please indicate the approximate proportion of light bulbs which are energy saving (or CFL)?  INT:READ OUT'

# idx = survey_raw[Q4905] <= 3
idx = survey_raw[Q4905] <= 2
survey['Q6'] = 0
survey.loc[idx, 'Q6'] = 0
# idx = survey_raw[Q4905] >= 4
idx = survey_raw[Q4905] >= 3
survey.loc[idx, 'Q6'] = 1

#%% Question 7
"""
House type
Free : 0
Connected: 1
"""

Q4905 = 'Question 450: I would now like to ask some questions about your home.  Which best describes your home?'
survey['Q7'] = -1

survey.loc[survey_raw[Q4905] == 3, 'Q7'] = 0
survey.loc[survey_raw[Q4905] == 5, 'Q7'] = 0

survey.loc[survey_raw[Q4905] == 2, 'Q7'] = 1
survey.loc[survey_raw[Q4905] == 4, 'Q7'] = 1

# others
survey.loc[(survey_raw[Q4905] <2).values * (survey_raw[Q4905] >5).values,'Q7'] = -1

#%% Question 10
"""
Type of cooking facility
Electric: 1
non-electric: 0
"""
Q4704 = 'Question 4704: Which of the following best describes how you cook in your home'
survey['Q10'] = 0
idx = survey_raw[Q4704] == 1
survey.loc[idx, 'Q10'] = 1

#%% Question 11
"""
Number of bedrooms
very low (0): <=2
low (1): ==3
high (2): ==4
very high(3): > 4
"""
Q460 = 'Question 460: How many bedrooms are there in your home'
survey['Q11'] = 0

# very high
idx = survey_raw[Q460] > 4
survey.loc[idx, 'Q11'] = 3

# high
idx = survey_raw[Q460] == 4
survey.loc[idx, 'Q11'] = 2

# low
idx = survey_raw[Q460] == 3
survey.loc[idx, 'Q11'] = 1

# very low
idx = survey_raw[Q460] <= 2
survey.loc[idx, 'Q11'] = 0

bad_idx = survey_raw[Q460] == 6
survey.loc[bad_idx, 'Q11'] = -1

#%% Question 12: 나중에,,
"""
Number of appliances
low (<=8)
med (8< <=11)
high (>11)
"""
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
for Question in Q49002:
    val = survey_raw.loc[:,Question].values.copy()
    val[val<=1] = 0
    val[val == 2] = 1
    val[val == 3] = 2
    val[val == 4] = 3
    survey['Q12'] += val

for Question in Q490002:
    val = survey_raw.loc[:,Question].values.copy()
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

#%% Questions 14
"""
0: AB
1: C1 C2
2: DE
"""
Q401 = 'Question 401: SOCIAL CLASS Interviewer, Respondent said that occupation of chief income earner was.... <CLASS> Please code'
survey['Q14'] = -1
idx = survey_raw[Q401] == 1
survey.loc[idx, 'Q14'] = 0

idx = (survey_raw[Q401] == 2).values + (survey_raw[Q401] == 3).values
survey.loc[idx, 'Q14'] = 1

idx = (survey_raw[Q401] == 4).values
survey.loc[idx, 'Q14'] = 2

#%% Questions 15
"""
6시간 이상 unoccupied: 1
others: 0
"""
Q430 = 'Question 430: And how many of these are typically in the house during the day (for example for 5-6 hours during the day)'
Q4312 = 'Question 4312: And how many of these are typically in the house during the day (for exanmple for 5-6 hours during the day)'

survey['Q15'] = 0
idx_1 = survey_raw[Q430] == 8
idx_2 = survey_raw[Q4312] == 8
survey.loc[idx_1, 'Q15'] = 1
survey.loc[idx_2, 'Q15'] = 1

#%% sort int Q
sorted_col = ['Q'+str(i) for i in range(1,16)]
survey = survey.reindex(sorted_col, axis=1)

#%% save
survey.to_csv('../data/CER/survey_processed_0728.csv',index=True)