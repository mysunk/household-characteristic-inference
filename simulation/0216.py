import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'size': 14, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

#%% 기존 논문들 (baseline)
result = pd.read_excel('etc/accuracy_0216.xlsx')
for qnum in range(15):
    plt.figure(figsize=(7,3))
    (result.iloc[qnum,3:] * 100).plot.barh(color=["tab:blue", "tab:blue",
                                                  "tab:orange", "tab:blue"])
    plt.title(result.iloc[qnum,2] + f'\n(# classes: {result.iloc[qnum,1]})')
    plt.xlim(np.min(result.iloc[qnum,3:] * 100) * 0.95,
             np.max(result.iloc[qnum,3:] * 100) * 1.05)
    plt.xlabel('Accuracy [%]')
    plt.tight_layout()
    plt.show()
