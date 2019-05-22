import os
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
from scipy import signal
from numpy import hstack
import spaceweather.analysis.gen_data as gen_data
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat
import spaceweather.rcca as rcca

import spaceweather.visualisation.heatmaps as svh


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





# def angle(ts1,ts2):
#     # input 2 time series, of the same length
#     # each data point in the time series should be a 3-dimensional vector with N, E and Z components
#     # return np.rad2deg(subspace_angles(bbb_a, bbb_b))
#      return np.rad2deg(subspace_angles(ts1, ts2))



ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",MLT = True, MLAT = True)
ds2 = sad.csv_to_Dataset(csv_file = "Data/20010305-16-38-supermag.csv",MLT = True, MLAT = True)

my_ds = sad.csv_to_Dataset(csv_file = "Data/20190521-14-08-supermag.csv",MLT = True, MLAT = True)


components=['N', 'E', 'Z']
# ds = data_funcs.mag_detrend(ds1)

scratch = sac.cca(ds1)

# scratch.loc[dict(first_st = 'TAL')]


# write a function that takes two time series and generates a graph showing how the angles between a and b vary over time
def angles(ts1, ts2):
    # ts1 and ts2 are Datasets

    scratch = sac.cca(ds1)
    return

scratch_threshold = sat.threshold(ds1)

sat.adj_mat(ds1,plot=True,thr_xrds=scratch_threshold)




svh.correlogram(my_ds,lag_range=10, win_len=128,ret=True, station1='NAL', station2='NRD')


svh.correlogram(my_ds,lag_range=10, win_len=128,ret=True, station1='NAL', station2='BJN')

svh.correlogram(my_ds,lag_range=10, win_len=128,ret=True, station1='NAL', station2='SOR')

svh.correlogram(my_ds,lag_range=10, win_len=128,ret=True)

