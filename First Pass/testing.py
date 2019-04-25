# need to be in RSG-Space-Weather folder
pwd()

## Packages
import numpy as np
import pandas as pd
import scipy.signal as scg
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import sys
import os
import lib.supermag as sm
import lib.rcca as rcca
# import xscale.signal.fitting as xsf # useful functions for xarray data structures
    # pip3 install git+https://github.com/serazing/xscale.git
    # pip3 install toolz

################################################################################
########## Restructuring the SuperMAG Data ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
ds1

ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            components = ['N', 'E', 'Z'],
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
################################################################################




################################################################################
########## Plotting ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
sm.plot_mag_data(ds=ds1)


## extra code for editing titles of plots
ds1 = ds1.loc[dict(station = slice('BLC'))]
stations = ds1.station.loc[dict(station = slice('BLC'))].values
components = ds1.component.values
## all of below code must be run simultaneously
g = ds1.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1, add_legend=False)
for i, ax in enumerate(g.axes.flat):
   ax.set_title(stations[i], fontsize=30)

plt.legend(labels=components, loc='right', title='Component', title_fontsize='x-large', fontsize=20)
plt.draw()
################################################################################




################################################################################
########## Detrending ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

det = sm.mag_detrend(ds=ds1)
det
################################################################################




################################################################################
########## Windowing ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

ds1_win = sm.window(ds = ds1)
ds1_win
ds1_win_60 = sm.window(ds = ds1, win_len = 60)
ds1_win_60
ds1_win_slice = sm.window(ds = ds1[dict(time=slice(0,10))], win_len = 3)
ds1_win_slice.measurements.loc[dict(station = 'TAL')]
ds1_win_slice[dict(window = 0)]
################################################################################




################################################################################
########## Canonical Correlation Analysis ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

## CCA between stations
test_inter = sm.inter_st_cca(ds = ds1)
test_inter.cca_coeffs

## CCA between components in one station
test_intra = sm.intra_st_cca(ds = ds1, station = 'BSL')
test_intra

## CCA between components for all stations
test_all = sm.st_cca(ds = ds1)
test_all
################################################################################




################################################################################
########## Thresholding ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

#--- Canonical Correlation ---
## KF Threshold
thresh_kf = sm.mag_thresh_kf(ds = ds1)
thresh_kf.thresholds.values

## Dods-style Threshold
thresh_dods = sm.mag_thresh_dods(ds = ds1, n0 = 0.25)
thresh_dods.thresholds.values

#--- Phase Correlation ---
## KF Threshold

## Dods-style Threshold

################################################################################




################################################################################
########## Constructing the Network ##########
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
ds1 = ds1[dict(time=slice(0,148))]

con_ds1 = sm.construct_network(ds = ds1, win_len = 128, n0 = 0.25)
con_ds2 = sm.construct_network(ds = ds1)
################################################################################




################################################################################
########## Visualizing the Network ##########

################################################################################
