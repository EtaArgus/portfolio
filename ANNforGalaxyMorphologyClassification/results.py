import glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
%matplotlib inline

# get data file names
path = r'C:\Users\Charles\Documents\SOR\Dissertation\Model\Results'
filenames = glob.glob(path + "/*.csv")
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Store file names
os.chdir('C:\\Users\Charles\Documents\SOR\Dissertation\Model\Results')
files = glob.glob("*.csv")

# Concatenate all data into one DataFrame
res = pd.concat(dfs, ignore_index=True)
res['Sample'] = np.zeros((len(res), 1))
res['Epoch'] = np.zeros((len(res), 1))
files
# Add column with sample name
for i in range(0, 12):
    for j in range(0 + i * 50, 50 + i * 50):
        res.loc[j, 'Sample'] = files[i]
        res.loc[j, 'Epoch'] = j - i * 50 + 1

# Change Column name
res = res.rename(columns={'Global': 'Accuracy'})


# 0-gz1-1
r1 = res[res['Sample'] == files[0]].sort_values(['Accuracy'], ascending=True)
r1['Epoch'] = range(0, 50)
res1 = sns.lmplot(x='Epoch', y='Accuracy', data=r1, fit_reg=False,
                  scatter_kws={"s": 20})
res1.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\gz1-0-1.pdf",
             bbox_inches='tight')
    maxi = np.max(res[res['Sample'] == files[11]].sort_values(['Accuracy'],
                                                              ascending=True))

# 0-gz1-2
r2 = res[res['Sample'] == files[1]].sort_values(['Accuracy'], ascending=True)
r2['Epoch'] = range(0, 50)
res2 = sns.lmplot(x='Epoch', y='Accuracy', data=r2, fit_reg=False,
                  scatter_kws={"s": 20})
res2.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\gz1-0-2.pdf",
             bbox_inches='tight')
max = np.max(res[res['Sample'] == files[3]].sort_values(['Accuracy'],
                                                        ascending=True))

# 0.8-gz1-1
r3 = res[res['Sample'] == files[4]].sort_values(['Accuracy'], ascending=True)
r3['Epoch'] = range(0, 50)
res3 = sns.lmplot(x='Epoch', y='Accuracy', data=r3, fit_reg=False,
                  scatter_kws={"s": 20})
res3.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\gz1-0.8-1.pdf",
             bbox_inches='tight')
max = np.max(res[res['Sample'] == files[6]].sort_values(['Accuracy'],
                                                        ascending=True))

# 0.8-gz1-2
r4 = res[res['Sample'] == files[5]].sort_values(['Accuracy'], ascending=True)
r4['Epoch'] = range(0, 50)
res4 = sns.lmplot(x='Epoch', y='Accuracy', data=r4, fit_reg=False,
                  scatter_kws={"s": 20})
res4.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\gz1-0.8-2.pdf",
             bbox_inches='tight')
max = np.max(res[res['Sample'] == files[7]].sort_values(['Accuracy'],
                                                        ascending=True))

# 0.95-gz1-1
r5 = res[res['Sample'] == files[8]].sort_values(['Accuracy'], ascending=True)
r5['Epoch'] = range(0, 50)
res5 = sns.lmplot(x='Epoch', y='Accuracy', data=r5, fit_reg=False,
                  scatter_kws={"s": 20})
res5.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\gz1-0.95-1.pdf",
             bbox_inches='tight')
max = np.max(res[res['Sample'] == files[10]].sort_values(['Accuracy'],
                                                         ascending=True))

# 0.95-gz1-2
r6 = res[res['Sample'] == files[9]].sort_values(['Accuracy'], ascending=True)
r6['Epoch'] = range(0, 50)
res6 = sns.lmplot(x='Epoch', y='Accuracy', data=r6, fit_reg=False,
                  scatter_kws={"s": 20})
res6.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\gz1-0.95-2.pdf",
             bbox_inches='tight')
max = np.max(res[res['Sample'] == files[11]].sort_values(['Accuracy'],
                                                         ascending=True))
