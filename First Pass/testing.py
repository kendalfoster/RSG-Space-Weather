# need to be in RSG-Space-Weather folder
pwd()

###############################################################################
########## Restructuring and Plotting the SuperMAG Data ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm


## Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
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

## Import and Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

## CCA between stations
test_inter = sm.inter_st_cca(ds = ds1)
test_inter.cca_coeffs

## CCA between readings in one station
test_intra = sm.intra_st_cca(ds = ds1, station = 'BSL')
test_intra

## CCA between readings for all stations
test_all = sm.st_cca(ds = ds1)
test_all
###############################################################################




###############################################################################
########## Thresholding ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm
import lib.rcca as rcca

## Import and Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

## KF Threshold
thresh_kf = sm.mag_thresh_kf(ds = ds1)
thresh_kf.thresholds.values

## Dods-style Threshold
thresh_dods = sm.mag_thresh_dods(ds = ds1, n0 = 0.25)
thresh_dods.thresholds.values
###############################################################################




###############################################################################
########## Windowing ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm
import lib.rcca as rcca

## Import and Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)


ds2 = ds1.measurements.loc[dict(station = 'TAL')]
ds2 = ds2[dict(time=slice(0,10))]
ds2.values
ds2_roll = ds2.rolling(time=3).construct(window_dim='window').dropna('time')
ds2_roll
###############################################################################




###############################################################################
########## Converting Numpy Arrays to xarray Datasets ##########
george = np.ones(shape=(2,3,4))
george_ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading', 'station'], george)},
                       coords = {'time': [2011,2012],
                                 'reading': ['N', 'E', 'Z'],
                                 'station': ['TAL', 'BLC', 'EKS', 'BLS']})
george_ds
# each coord requires
    # 1) a name that matches a dimension name from data_vars
    # 2) a vector whose length matches the 'length' of the data along the specified dimension
###############################################################################
