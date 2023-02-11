# Inferring Socio-Demographic Information Using Smart Meter Data by Transfer Learning

This repository is the official implementation of Inferring Socio-Demographic Information Using Smart Meter Data by Transfer Learning.  
You can find the paper here:
[link](https://ieeexplore.ieee.org/document/9791982)


Introduction
=======================================
* This paper proposes a framework for inferring socio-demographic information using smart meter data
* Collecting household characteristics information and corresponding smart meter data requires considerable effort and cost
* We present a transfer learning methodology using datasets collected from different areas

Steps to Reproduce
==================
### Download raw dataset
Download following datasets  
`SAVE Dataset`: [link](https://beta.ukdataservice.ac.uk/datacatalogue/doi/?id=8676#1)  
`CER Dataset`: [link](http://www.ucd.ie/issda/data/commissionforenergyregulationcer/)

Data location should be like belows
```
.
|- data
    |- raw # raw datasets downloaded
        |- CER
        |- SAVE
```

### Install requirements
```
pip install -r requirements.txt
```

### Prepare data from raw dataset
Make time series energy dataset and label
```
python prepare/cer_dataset.py
python prepare/save_dataset.py
```
From this process, we can get dataset in `data/prepared` like below.
```
.
|- data
    |- raw # raw datasets downloaded
        |- CER
        |- SAVE
    |- prepared # prepared dataset
        |- CER
            |- energy.csv
            |- info.csv
        |- SAVE
            |- energy.csv
            |- info.csv
```
energy.csv sample
```
                     956600034  956600058  956600128
timestamp                                              
2018-01-01 00:00:00       0.006         NaN         NaN
2018-01-01 00:15:00       0.017         NaN         NaN
2018-01-01 00:30:00       0.007         NaN         NaN
2018-01-01 00:45:00       0.008         NaN         NaN
2018-01-01 01:00:00       0.016         NaN         NaN
```
info.csv sample
```
            Q1     Q2     Q3   Q4   Q5   Q6   Q7
BMG_ID                                          
956600035  3.0  False  False  0.0  NaN  0.0  1.0
956600128  4.0  False  False  1.0  NaN  1.0  1.0
956600093  1.0   True   True  2.0  NaN  0.0  0.0
956600107  2.0  False  False  2.0  NaN  0.0  1.0
956600090  2.0  False  False  1.0  NaN  1.0  1.0
```
`SAVE dataset`: 3938 households  
`CER dataset`: 3938 households

### Train and Evaluate
```
python main.py
```

Methodology
=======================================
![methodology](img/fig1.png)

### 1. Daily typical load generation
From time series dataset, get daily typical load by removing untypical sample
### 2. Instance selection
To prevent negative transfer remove instance which reduce similarity between two datasets by measuring [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
### 3. Feature selection
Extract influential features for both Source and Target datasets to remove noisy feature by [mrmr](https://pypi.org/project/pymrmr/)
### 4. Training
Train deep learning model with the source dataset
### 5. Fine tuning
Train the pre-trained model with target dataset
### 6. Testing
Infer test dataset with the rest of target dataset

Benchmarking
=======================================
Compare our methodology with benchmarking methodologys in the face of a shortage of datasets

1. @ paper:
Wang, Yi, et al. 
"Deep learning-based socio-demographic information identification from smart meter data."
IEEE Transactions on Smart Grid 10.3 (2018): 2593-2602.  
2. @ paper:
Yan, Siqing, et al. 
"Time–Frequency Feature Combination Based Household Characteristic Identification Approach Using Smart Meter Data." 
IEEE Transactions on Industry Applications 56.3 (2020): 2251-2262.
```
python benchmarking.py
python benchmarking-2.py
```

Contact
==================
If there is something wrong or you have any questions, send me an [![Gmail Badge](https://img.shields.io/badge/-Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:pond9816@gmail.com)](mailto:pond9816@gmail.com) or make an issue.  
