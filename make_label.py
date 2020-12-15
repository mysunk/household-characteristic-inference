import pandas as pd
import numpy as np

survey_raw = pd.read_csv('data/survey/Smart meters Residential pre-trial survey data.csv', encoding='cp1252', low_memory=False)
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
Qnum = '410'
np.where(np.array([Qnum in col for col in survey_raw.columns]))
survey_raw.columns[8]

#%% Questions 1
"""
Young (1): age <= 35
Ned (2): 35 < age <= 65
High (3): 65 < age
"""
Q300 = 'Question 300: May I ask what age you were on your last birthday? INT: IF NECCESSARY, PROMPT WITH AGE BANDS'
survey['Q1'] = -1

idx = survey_raw[Q300] == 6
survey.loc[idx, 'Q1'] = 3
idx = (survey_raw[Q300] < 6).values * (survey_raw[Q300] > 2).values
survey.loc[idx, 'Q1'] = 2
idx = (survey_raw[Q300] <= 2).values
survey.loc[idx, 'Q1'] = 1

#%% Questions 2
"""
retired: 1
not: 0
"""
Q310 = 'Question 310: What is the employment status of the chief income earner in your household, is he/she'
survey['Q2'] = 0

idx = survey_raw[Q310] == 6
survey.loc[idx, 'Q2'] = 1

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

# initialize
survey['Q13'] = -1
survey['Q3'] = -1
survey['Q8'] = 0
survey['Q9'] = 0

# 세대원 수가 1명인 경우
survey.loc[survey_raw[Q410] == 1, 'Q13'] = 1
survey.loc[survey_raw[Q410] == 1, 'Q3'] = 0

# 모든 세대원이 15살 이상인 경우
survey.loc[survey_raw[Q410] == 2, 'Q13'] = survey_raw.loc[survey_raw[Q410] == 2, Q420]
survey.loc[survey_raw[Q410] == 2, 'Q3'] = 0

# children이 있는 경우
survey.loc[survey_raw[Q410] == 3, 'Q13'] = survey_raw.loc[survey_raw[Q410] == 3, Q420] + survey_raw.loc[survey_raw[Q410] == 3, Q43111]
survey.loc[survey_raw[Q410] == 3, 'Q3'] = survey_raw.loc[survey_raw[Q410] == 3, Q43111]

# children을 binary로
survey.loc[survey['Q3'] >= 1,'Q3'] = 1

# is single?
idx = (survey['Q3'] == 0).values * (survey['Q13'] == 1).values
survey.loc[idx, 'Q8'] = 1

# is family?
idx = (survey['Q3'] >0 ).values * (survey['Q13'] > 1).values
survey.loc[idx, 'Q9'] = 1

# #residents를 3 class로
survey.loc[survey['Q13'] <= 2,'Q13'] = 1
survey.loc[survey['Q13'] > 2,'Q13'] = 2

#%% Question 4
"""
age of building
old (>30)
new (<=30)
"""
Q4531 = 'Question 4531: Approximately how old is your home?'
Q453 = 'Question 453: What year was your house built INT ENTER FOR EXAMPLE: 1981- CAPTURE THE FOUR DIGITS'
survey['Q4'] = -1
idx = (survey_raw[Q453] >= 1500) * (survey_raw[Q453] <= 2010)
survey.loc[idx, 'Q4'] = 2009 - survey_raw.loc[idx, Q453]
idx = (survey_raw[Q4531] != -1)
survey.loc[idx, 'Q4'] = survey_raw.loc[idx, Q4531]

idx = survey['Q4'] > 30
bad_idx = survey['Q4'] == -1
survey.loc[idx,'Q4'] = 1
survey.loc[~idx,'Q4'] = 0
survey.loc[bad_idx,'Q4'] = -1

#%% Question 5
"""
floar area
small: 1
med: 2
big: 3
"""

Q6103 = 'Question 6103: What is the approximate floor area of your home?'
Q61031 = 'Question 61031: Is that'

survey['Q5'] = survey_raw.copy()

# bad data
bad_idx = survey_raw[Q6103] == 999999999
survey.loc[bad_idx, 'Q5'] = -1

idx = survey_raw[Q61031] == 2
survey.loc[idx, 'Q5'] /= 0.092903

idx = survey_raw[Q6103] <= 100
idx3 = survey_raw[Q6103] > 200
idx2 = (survey_raw[Q6103] > 100).values * (survey_raw[Q6103] <= 200).values

survey.loc[idx,'Q5'] = 1
survey.loc[idx2,'Q5'] = 2
survey.loc[idx3,'Q5'] = 3
survey.loc[bad_idx,'Q5'] = -1
survey['Q5'] = survey['Q5'].astype(int)

#%% Question 6
"""
Energy-efficient light bulb proportion
1: up to halp
2: about 3 quarters of more
"""
Q4905 = 'Question 4905: And now considering energy reduction in your home please indicate the approximate proportion of light bulbs which are energy saving (or CFL)?  INT:READ OUT'

idx = survey_raw[Q4905] <= 3
survey['Q6'] = 0
survey.loc[idx, 'Q6'] = 1
idx = survey_raw[Q4905] >= 4
survey.loc[idx, 'Q6'] = 2

#%% Question 7
"""
House type
Free : 1
Connected: 2
"""

Q4905 = 'Question 450: I would now like to ask some questions about your home.  Which best describes your home?'
survey['Q7'] = -1

survey.loc[survey_raw[Q4905] == 3, 'Q7'] = 1
survey.loc[survey_raw[Q4905] == 5, 'Q7'] = 1

survey.loc[survey_raw[Q4905] == 2, 'Q7'] = 2
survey.loc[survey_raw[Q4905] == 4, 'Q7'] = 2

# others
survey.loc[survey['Q7'] == 0,'Q7'] = -1

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
very low (1): <=2
low (2): ==3
high (3): ==4
very high(4): > 4
"""
Q460 = 'Question 460: How many bedrooms are there in your home'
survey['Q11'] = 0

# very high
idx = survey_raw[Q460] > 4
survey.loc[idx, 'Q11'] = 4

# high
idx = survey_raw[Q460] == 4
survey.loc[idx, 'Q11'] = 3

# low
idx = survey_raw[Q460] == 3
survey.loc[idx, 'Q11'] = 2

# very low
idx = survey_raw[Q460] <= 2
survey.loc[idx, 'Q11'] = 1

bad_idx = survey_raw[Q460] == 6
survey.loc[bad_idx, 'Q11'] = -1

#%% Question 12: 나중에,,
"""
Number of appliances
low (<=8)
med (8< <=11)
high (>11)
"""

for idx in np.where(np.array([Qnum in col for col in survey_raw.columns]))[0]:
    print(idx)
    print(survey_raw.columns[idx])

#%% Questions 14
"""
1: AB
2: C1 C2
3: DE
"""
Q401 = 'Question 401: SOCIAL CLASS Interviewer, Respondent said that occupation of chief income earner was.... <CLASS> Please code'
survey['Q14'] = -1
idx = survey_raw[Q401] == 1
survey.loc[idx, 'Q14'] = 1

idx = (survey_raw[Q401] == 2).values + (survey_raw[Q401] == 3).values
survey.loc[idx, 'Q14'] = 2

idx = (survey_raw[Q401] == 4).values
survey.loc[idx, 'Q14'] = 3

#%% Questions 15
"""
6시간 이상 unoccupied: 1
others: 0
"""
Q430 = 'Question 430: And how many of these are typically in the house during the day (for example for 5-6 hours during the day)'
survey['Q15'] = 0
survey.loc[survey_raw[Q430] == 8,'Q15'] = 1
