import data_loader as dl
import Network2
import glob
import pandas as pd
import os

# import numpy as np
gz1 = dl.prepare_data_gz1(0.95, nrows=700000)
gz2 = dl.prepare_data_gz2(0.9, 't01')


tr_1, tt_1 = dl.data_wrapper(gz1, nvar=5, nclass=3)
tr_2, tt_2 = dl.data_wrapper(gz2, nvar=5, nclass=3)

net = Network2.Network([5, 10, 10, 3])
net.SGD(tr_1, 50, 10, 0.1, '0.95-gz2-2.csv', tt_2)

# get data file names
path = r'C:\Users\Charles\Documents\SOR\Dissertation\Model\Results'
filenames = glob.glob(path + "/*.csv")
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))
os.chdir('C:\\Users\Charles\Documents\SOR\Dissertation\Model\Results')
files = glob.glob("*.csv")


type(dfs[1][1])

for i in dfs:
    for j in dfs[i]:
        dfs[i].iloc


# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
