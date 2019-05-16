import os
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
from scipy import signal
from numpy import hstack
import lib.generate_model_data as gen_data
import lib.supermag as sm
import lib.rcca as rcca

scratch_ds = gen_data.generate_one_day_time_series('2001-04-03', '08:00:00', 30, 4, [0, 0.25, 0.5],['XXX','YYY'])

scratch_ds

scratch_N = scratch_ds.measurements.loc[:,'N','YYY']
scratch_E = scratch_ds.measurements.loc[:,'E','YYY']
scratch_Z = scratch_ds.measurements.loc[:,'Z','YYY']

scratch_ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)


scratch_ds.measurements[480:510,:,:].plot.line(x='time', hue='component', col='station', col_wrap=1)


def inter_phase_dir_corr(ds,station1,station2,wind_start1,wind_start2,readings=None):
     #check if readings are provided
     if readings is None:
         readings = ['N', 'E', 'Z']

     # universally necessary things
     num_read = len(readings)

     # setup row array for the correlation coefficients
     cca_coeffs = np.zeros(shape = (1, num_read), dtype = float)

     # get readings for the station
     data = sm.window(ds,128)
     data1 = data.measurements.loc[dict(station = station1)][dict(win_start = wind_start1)]
     data2 = data.measurements.loc[dict(station = station2)][dict(win_start = wind_start2)]


#data1 = data.measurements.loc[dict(station = station1)].loc[dict(win_start = wind[wind_start1])]
#data2 = data.measurements.loc[dict(station = station2)].loc[dict(win_start = wind[wind_start2])]
     #Calculate the cannonical correlation between the directional meaurements on each station
     for i in range(num_read):
         first_read = data1[:,i].data
         first_read = np.reshape(first_read, newshape=[len(first_read),1])

         second_read = data2[:,i].data
         second_read = np.reshape(second_read, newshape=[len(second_read),1])

         temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
         cca_coeffs[0,i] = abs(temp_cca.train([first_read, second_read]).cancorrs[0])
     return cca_coeffs


##### Function that finds the index of the point where the phases are at their highest
#     correlation
#     input: ds - dataset output from mag_csv_to_Dataset Function
#           station 1 - 3 letter code for the station 1 as a string, ex: 'BLC'
#           station 2 - 3 letter code for the station 2 as a string, ex: 'BLC'
#           wind_start1 - Index of start of both winow
#    output: shift - the amount of shfit needed to move station 2's window inline


def phase_finder(ds, station1, station2, start):

    ### Get the data windows
    data = sm.window(ds,128)
    data1 = data.measurements.loc[dict(station = station1)]
    data2 = data.measurements.loc[dict(station = station2)]

    ## Set up matrix to put our correlation parameters into
    corr_coeff = np.zeros(shape = (21), dtype = float)


    ## Shift the second window amongst the first one and caluclate mean
    ## of the correlation readings for each shift
    for i in range(21):
         wind2 = start - 10 + i
         x = inter_phase_dir_corr(ds,station1,station2,start,wind2)
         corr_coeff[i] = np.mean(x)

    ## Find where the correlations are highest
    s = np.where(corr_coeff == np.amax(corr_coeff))
    shift = -10 + s[0][0]
    return shift



phase_finder(scratch_ds, "XXX", "YYY", 480)

sm.window(scratch_ds)


def angle(ts1,ts2):
    # input 2 time series, of the same length
    # each data point in the time series should be a 3-dimensional vector with N, E and Z components
    # return np.rad2deg(subspace_angles(bbb_a, bbb_b))
     return np.rad2deg(subspace_angles(ts1, ts2))
