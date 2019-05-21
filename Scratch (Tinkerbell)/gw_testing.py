import os
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
from scipy import signal
from numpy import hstack
import spaceweather.analysis.gen_data as gen_data
import spaceweather.analysis.cca as cca
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat
import spaceweather.rcca as rcca

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

components=['N', 'E', 'Z']
# ds = data_funcs.mag_detrend(ds1)

scratch = cca.cca(ds1,components)

scratch


scratch.comps.loc[dict(first_st = 'TAL', second_st = 'BLC', uv = 'u')]

True in np.isnan([np.nan,1,2,3])


# universal constants
stations = ds.station.values
num_st = len(stations)
num_ws = len(components)
num_cp = len(ds.time)

# setup (symmetric) arrays for each attribute
coeffs_arr = np.zeros(shape = (num_st, num_st), dtype = float)
weights_arr = np.zeros(shape = (num_st, num_st, 2, num_ws), dtype = float)
ang_rel_arr = np.zeros(shape = (num_st, num_st), dtype = float)
ang_abs_arr = np.zeros(shape = (num_st, num_st, num_cp, 2), dtype = float)
comps_arr = np.zeros(shape = (num_st, num_st, 2, num_cp), dtype = float)


i=4
j=5
st_1 = ds.measurements.loc[dict(station = stations[i])]
st_2 = ds.measurements.loc[dict(station = stations[j])]

comb_st = xr.concat([st_1, st_2], dim = 'component')
comb_st_no_na = comb_st.dropna(dim = 'time', how = 'any')
st_1 = comb_st[:, 0:num_ws]
st_2 = comb_st[:, num_ws:2*num_ws]
st_1_no_na = comb_st_no_na[:, 0:num_ws]
st_2_no_na = comb_st_no_na[:, num_ws:2*num_ws]


nan_times = []
for scratch_time in comb_st.time.values:
    if np.isnan(sum(comb_st.loc[dict(time = scratch_time)].values)):
        nan_times.append(scratch_time)

temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
ccac = temp_cca.train([st_1_no_na, st_2_no_na])
## store cca attributes ##
# coeffs
coeffs_arr[i,j] = ccac.cancorrs[0]
coeffs_arr[j,i] = coeffs_arr[i,j] # mirror results
# weights (a and b from Wikipedia)
w0 = ccac.ws[0].flatten() # this is a
w1 = ccac.ws[1].flatten() # this is b
weights_arr[i,j,0,:] = w0
weights_arr[i,j,1,:] = w1
weights_arr[j,i,0,:] = w0 # mirror results
weights_arr[j,i,1,:] = w1 # mirror results
# angles, relative
wt_norm = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(w1**2))
ang_rel_arr[i,j] = np.rad2deg(np.arccos(np.clip(np.dot(w0, w1)/wt_norm, -1.0, 1.0)))
ang_rel_arr[j,i] = ang_rel_arr[i,j] # mirror results
# angles, absolute

for k,timestamp in enumerate(ds.time):
    if timestamp in nan_times:
        print("here")
        ang_abs_arr[i,j,k,0] = np.nan
        ang_abs_arr[i,j,k,1] = np.nan
        ang_abs_arr[j,i,k,0] = np.nan
        ang_abs_arr[j,i,k,1] = np.nan
    else:
        xdata = st_1[dict(time=k)].values
        ydata = st_2[dict(time=k)].values
        wt_nrm0 = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(xdata**2))
        wt_nrm1 = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(ydata**2))
        print(np.rad2deg(np.arccos(np.clip(np.dot(w0, xdata)/wt_nrm0, -1.0, 1.0))))
        ang_abs_arr[i,j,k,0] = np.rad2deg(np.arccos(np.clip(np.dot(w0, xdata)/wt_nrm0, -1.0, 1.0)))
        ang_abs_arr[i,j,k,1] = np.rad2deg(np.arccos(np.clip(np.dot(w1, ydata)/wt_nrm1, -1.0, 1.0)))
        ang_abs_arr[j,i,k,0] = ang_abs_arr[i,j,k,0]
        ang_abs_arr[j,i,k,1] = ang_abs_arr[i,j,k,1]



# comps (a^T*X and b^T*Y from Wikipedia)
comps_arr[i,j,0,:] = ccac.comps[0].flatten() # this is a^T*X
comps_arr[i,j,1,:] = ccac.comps[1].flatten() # this is b^T*Y
comps_arr[j,i,0,:] = comps_arr[i,j,0,:] # mirror results
comps_arr[j,i,1,:] = comps_arr[i,j,1,:] # mirror results


# shrinking nested for loops to get all the pairs of stations
for i in range(0, num_st-1):
    st_1 = ds.measurements.loc[dict(station = stations[i])]
    for j in range(i+1, num_st):
        st_2 = ds.measurements.loc[dict(station = stations[j])]
        # remove NaNs from data (will mess up cca)
        comb_st = xr.concat([st_1, st_2], dim = 'component')
        comb_st_no_na = comb_st.dropna(dim = 'time', how = 'any')
        st_1 = comb_st[:, 0:num_ws]
        st_2 = comb_st[:, num_ws:2*num_ws]
        st_1_no_na = comb_st_no_na[:, 0:num_ws]
        st_2_no_na = comb_st_no_na[:, num_ws:2*num_ws]
        nan_times = []
        for scratch_time in comb_st.time.values:
            if np.isnan(sum(comb_st.loc[dict(time = scratch_time)].values)):
                nan_times.append(scratch_time)

        # run cca, suppress rcca output
        temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
        ccac = temp_cca.train([st_1_no_na, st_2_no_na])
        ## store cca attributes ##
        # coeffs
        coeffs_arr[i,j] = ccac.cancorrs[0]
        coeffs_arr[j,i] = coeffs_arr[i,j] # mirror results
        # weights (a and b from Wikipedia)
        w0 = ccac.ws[0].flatten() # this is a
        w1 = ccac.ws[1].flatten() # this is b
        weights_arr[i,j,0,:] = w0
        weights_arr[i,j,1,:] = w1
        weights_arr[j,i,0,:] = w0 # mirror results
        weights_arr[j,i,1,:] = w1 # mirror results
        # angles, relative
        wt_norm = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(w1**2))
        ang_rel_arr[i,j] = np.rad2deg(np.arccos(np.clip(np.dot(w0, w1)/wt_norm, -1.0, 1.0)))
        ang_rel_arr[j,i] = ang_rel_arr[i,j] # mirror results
        # angles, absolute

        for k,timestamp in enumerate(ds.time):
            if timestamp in nan_times:
                ang_abs_arr[i,j,k,0] = np.nan
                ang_abs_arr[i,j,k,1] = np.nan
                ang_abs_arr[j,i,k,0] = np.nan
                ang_abs_arr[j,i,k,1] = np.nan
            else:
                xdata = st_1[dict(time=k)].values
                ydata = st_2[dict(time=k)].values
                wt_nrm0 = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(xdata**2))
                wt_nrm1 = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(ydata**2))
                ang_abs_arr[i,j,k,0] = np.rad2deg(np.arccos(np.clip(np.dot(w0, xdata)/wt_nrm0, -1.0, 1.0)))
                ang_abs_arr[i,j,k,1] = np.rad2deg(np.arccos(np.clip(np.dot(w1, ydata)/wt_nrm1, -1.0, 1.0)))
                ang_abs_arr[j,i,k,0] = ang_abs_arr[i,j,k,0]
                ang_abs_arr[j,i,k,1] = ang_abs_arr[i,j,k,1]
        # comps (a^T*X and b^T*Y from Wikipedia)
        comps_arr[i,j,0,:] = ccac.comps[0].flatten() # this is a^T*X
        comps_arr[i,j,1,:] = ccac.comps[1].flatten() # this is b^T*Y
        comps_arr[j,i,0,:] = comps_arr[i,j,0,:] # mirror results
        comps_arr[j,i,1,:] = comps_arr[i,j,1,:] # mirror results

# build Dataset from coeffs
coeffs = xr.Dataset(data_vars = {'coeffs': (['first_st', 'second_st'], coeffs_arr)},
                    coords = {'first_st': stations,
                              'second_st': stations})
# build Dataset from weights
weights = xr.Dataset(data_vars = {'weights': (['first_st', 'second_st', 'ab', 'component'], weights_arr)},
                     coords = {'first_st': stations,
                               'second_st': stations,
                               'ab': ['a', 'b'],
                               'component': components})
# build Dataset from angles
ang_rel = xr.Dataset(data_vars = {'ang_rel': (['first_st', 'second_st'], ang_rel_arr)},
                     coords = {'first_st': stations,
                               'second_st': stations})
ang_abs = xr.Dataset(data_vars = {'ang_abs': (['first_st', 'second_st', 'index', 'ab'], ang_abs_arr)},
                     coords = {'first_st': stations,
                               'second_st': stations,
                               'index': range(num_cp),
                               'ab': ['a', 'b']})
# build Dataset from comps
comps = xr.Dataset(data_vars = {'comps': (['first_st', 'second_st', 'uv', 'index'], comps_arr)},
                   coords = {'first_st': stations,
                             'second_st': stations,
                             'uv': ['u', 'v'],
                             'index': range(num_cp)})

# merge Datasets
res = xr.merge([coeffs, weights, ang_rel, ang_abs, comps])
