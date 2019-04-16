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

ct = 0.7
ct_mat = sm.inter_st_cca(ds=ds1)
ct_mat.cca_coeffs.values
ct_mat.where(ct_mat > ct, 0).cca_coeffs.values

def mag_thresh(ds, n0=0.25, readings=['N', 'E', 'Z']):
    ct_mat = sm.inter_st_cca(ds=ds, readings=readings)
    ct_vec = np.linspace(start=0, stop=1, num=101)

    # iterate through all possible ct values
    res = ct_mat.where(ct_mat > ct_vec[0], 0)
    for i in ct_vec[1:]:
        res = xr.concat([res, ct_mat.where(ct_mat > i, 0)], dim = 'C_T')
        # sum over row and column to get degrees of each station

    res = res.assign_coords(C_T = ct_vec)

res.cca_coeffs.loc[0.75].values

## Alternate Method ##
# for each possible threshold, C_T
    # for each time in times
    # calculate inter_st_cca with universal threshold, C_T
    # stack to get 3-dimensional Dataset
    # take mean over time dimension for each (i,j) pair in the CCA matrix
        # mean is sum((i,j) pair over dim='time') / (num_st - 1)
        # (now it's back to 2 dims)
    # stack the (now) 2 dim Dataset to make a 3-dim Dataset
        # (3rd dim is values of C_T)
# now we have Dataset of threshold values, where dims = first_st, second_st, C_T
    # for every station pair (i,j)
    # find 'C_T' index whose value equals n_0
    # use that value of C_T for the station pair (i,j)
    # store these threshold values in a 2-dim Dataset?
###############################################################################
