import os
import sys
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_style("darkgrid")

# Set working directory
os.chdir('C:\\Users\Charles\Documents\SOR\Dissertation\Model')


data = gz2
list(data)
# Correlation matrix
corrmat = data.ix[:, ['dered_g-dered_r', 'dered_r-dered_i',
                      'deVAB_i', 'expAB_i', 'lnLexp_i', 'lnLdeV_i', 'lnLstar_i',
                      'mRrCc_i', 'mCr4_i', 'texture_i', 'concentration']].corr()
fig, ax = plt.subplots()
ax = sns.heatmap(corrmat, cbar=True, annot=True, vmax=.8, square=True,
                 annot_kws={'size': 8}, fmt='.2f', rasterized=True,
                 cmap="Reds")
fig.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\cm.pdf",
            bbox_inches='tight')

sp1 = sns.lmplot('dered_g-dered_r', 'deVAB_i', data=data,
                 hue='GROUP', fit_reg=False, scatter_kws={"s": 0.5}, aspect=1)
sp1.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\sp1sa.png",
            bbox_inches='tight')

sp2 = sns.lmplot('dered_g-dered_r', 'concentration',
                 data=data[(data['concentration'] < 5)],
                 hue='GROUP', fit_reg=False, scatter_kws={"s": 0.5}, aspect=1)
sp2.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\sp2sa.png",
            bbox_inches='tight')

sp3 = sns.lmplot('dered_g-dered_r', 'mRrCc_i',
                 data=data[(data['mRrCc_i'] < 700)], hue='GROUP',
                 fit_reg=False, scatter_kws={"s": 0.5}, aspect=1)
sp3.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\sp3sa.png",
            bbox_inches='tight')

sp4 = sns.lmplot('deVAB_i', 'concentration',
                 data=data[(data['concentration'] < 4)],
                 hue='GROUP', fit_reg=False, scatter_kws={"s": 0.5}, aspect=1)
sp4.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\sp4sa.png",
            bbox_inches='tight')

sp5 = sns.lmplot('deVAB_i', 'mRrCc_i',
                 data=data[(data['mRrCc_i'] < 300)],
                 hue='GROUP', fit_reg=False, scatter_kws={"s": 0.5}, aspect=1)
sp5.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\sp5sa.png",
            bbox_inches='tight')

sp6 = sns.lmplot('concentration', 'mRrCc_i',
                 data=data[(data['mRrCc_i'] < 400) & (data['concentration'] < 5)],
                 hue='GROUP', fit_reg=False, scatter_kws={"s": 0.5}, aspect=1)
sp6.savefig("C:\\Users\Charles\Documents\SOR\Dissertation\Report\sp6sa.png",
            bbox_inches='tight')

# distributions

# dered_g-dered_r description
plt.plot(data['dered_g-dered_r'])
d1 = sns.kdeplot(data[data['GROUP'] == 0]['dered_g-dered_r'],
                 label='Spiral', shade=True)
sns.kdeplot(data[data['GROUP'] == 1]['dered_g-dered_r'],
            label='Elliptical', shade=True)
sns.kdeplot(data[data['GROUP'] == 2]['dered_g-dered_r'],
            label='Star/Artefac', shade=True)
d1.axis(ymin=0, ymax=2.7)
sns.boxplot(x='GROUP', y='dered_g-dered_r',
            data=pd.concat([data['dered_g-dered_r'], data['GROUP']], axis=1))

# de Vaucouleurs fit axial ratio
plt.plot(data['deVAB_i'])
d2 = sns.kdeplot(data[data['GROUP'] == 0]['deVAB_i'],
                 label='Spiral', shade=True)
sns.kdeplot(data[data['GROUP'] == 1]['deVAB_i'],
            label='Elliptical', shade=True)
sns.kdeplot(data[data['GROUP'] == 2]['deVAB_i'],
            label='Star/Artefac', shade=True)
d2.axis(ymin=0, ymax=2)
sns.boxplot(x='GROUP', y='deVAB_i',
            data=pd.concat([data['deVAB_i'], data['GROUP']], axis=1))

# Concentration
plt.plot(data['concentration'])
d3 = sns.kdeplot(data[data['GROUP'] == 0]['concentration'],
                 label='Spiral', shade=True)
sns.kdeplot(data[data['GROUP'] == 1]['concentration'],
            label='Elliptical', shade=True)
sns.kdeplot(data[data['GROUP'] == 2]['concentration'],
            label='Star/Artefac', shade=True)
d3.axis(xmin=1, xmax=4, ymin=0, ymax=1.25)
sns.boxplot(x='GROUP', y='concentration',
            data=pd.concat([data['concentration'], data['GROUP']], axis=1))

# mCr4_i
plt.plot(data['mCr4_i'])
d4, ax = plt.subplots()
d4 = sns.kdeplot(data[data['GROUP'] == 0]['mCr4_i'],
                 label='Spiral', shade=True)
sns.kdeplot(data[data['GROUP'] == 1]['mCr4_i'],
            label='Elliptical', shade=True)
sns.kdeplot(data[data['GROUP'] == 2]['mCr4_i'],
            label='Star/Artefac', shade=True)
d4.axis(xmin=1.8, xmax=2.8, ymin=0, ymax=6)
sns.boxplot(x='GROUP', y='mCr4_i',
            data=pd.concat([data['mCr4_i'], data['GROUP']], axis=1))
