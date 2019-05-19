# need to be in RSG-Space-Weather folder
pwd()

###############################################################################
########## Restructuring and Plotting the SuperMAG Data ##########
import numpy as np
import pandas as pd

import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm
# importing mean()
from statistics import mean
import seaborn as sns

## Restructure SuperMAG
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/moredata.csv", MLT = True, MLAT = True)

ds1

ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            readings = ['N', 'E', 'Z'],
                            MLT = True, MLAT = True)
ds2

ds3 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = False, MLAT = True)
ds3 # exclude MLT data

ds4 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = False)
ds4 # exclude MLAT data, order of stations should be different compared to above

ds4 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = False, MLAT = False)
ds4 # exclude MLT and MLAT data, order of stations should also be different

## Plot SuperMAG Data
sm.plot_mag_data(ds=ds1)
###############################################################################




###############################################################################
########## Canonical Correlation Analysis ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm
import lib.rcca as rcca



## CCA between stations
test_inter = sm.inter_st_cca(ds = ds1);
test_inter.cca_coeffs

## CCA between readings in one station
test_intra = sm.intra_st_cca(ds = ds1, station = 'BSL')
test_intra

## CCA between readings for all stations
test_all = sm.st_cca(ds = ds1)
test_all
###############################################################################
ds1 = sm.mag_detrend(ds2, type='linear')

ds = ds1
station1 = "BLC"
station2 = "BSL"
lag_range=7
win_len=400

#Window the data
windowed = sm.window(ds,win_len)
windowed

a = windowed.measurements.loc[dict(station = station1)].loc[dict(reading = "N")][:,0]
time_length = len(a)
time_range = time_length - 2 * lag_range
a.shape

x = np.arange(time_range) + lag_range + 1
y = np.arange(2*lag_range+1) - lag_range
z = np.zeros([len(y),time_range])


for i in range(len(y)):
    for j in range(time_range):
        corr = sm.inter_phase_dir_corr(ds,station1,station2,x[j]-1,y[i]+x[j]-1,win_len,readings=None)
        z[i,j] = np.mean(corr)


plot = sns.heatmap(z,vmin=0,vmax=1,yticklabels=y)


m = np.argmax(z,axis=0)
u = np.zeros([len(y),time_range])

for i in range(time_range):
    a = m[i]
    u[a,i] = 1

#Produce heatmap
plot = sns.heatmap(u,vmin=0,vmax=1,yticklabels=y)




a = sm.corellogram(ds, station1, station2, lag_range, win_len)
b = sm.corellogram_max(ds, station1, station2,lag_range,win_len)
