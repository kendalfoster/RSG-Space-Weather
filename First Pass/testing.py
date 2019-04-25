# need to be in RSG-Space-Weather folder
pwd()
import os
os.chdir('/home/yy/Documents/RSG-Space-Weather/')
os.getcwd()
###############################################################################
########## Restructuring and Plotting the SuperMAG Data ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm




data = pd.read_csv("First Pass/20190403-00-22-supermag.csv")
times = pd.to_datetime(data['Date_UTC'].unique())

#-----------------------------------------------------------------------
#---------- optional arguments -----------------------------------------
#-----------------------------------------------------------------------

# if MLAT is included, sort and make Dataset

# sort stations by magnetic latitude (from north to south)
stations = data['IAGA'].unique()
num_st = len(stations)
mlat_arr = np.vstack((stations,np.zeros(num_st))).transpose()
for i in range(0,num_st):
    mlat_arr[i,1] = data['MLAT'].loc[data['IAGA'] == stations[i]].mean()
mlat_arr = sorted(mlat_arr, key=lambda x: x[1], reverse=True)
stations = [i[0] for i in mlat_arr]
mlats = [round(i[1],4) for i in mlat_arr]
# build MLAT Dataset, for merging later
ds_mlat = xr.Dataset(data_vars = {'mlats': (['station'], mlats)},
                     coords = {'time': times,
                               'reading': readings,
                                   'station': stations})

data
stations
mlats
mlats[1]

type((['station'], mlats))
type(stations)


## Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
ds1

ds1.measurements.loc[dict(station = "BLC")]

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
ds1.measurements.plot.line(x='time', hue='station', col='reading', col_wrap=1)
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

## Import and Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

## CCA between stations
test_inter = sm.inter_st_cca(ds = ds1)
test_inter

## CCA between readings in one station
test_intra = sm.intra_st_cca(ds = ds1, station = 'BSL')
test_intra

## CCA between readings for all stations
test_all = sm.st_cca(ds = ds1)
test_all
###############################################################################
