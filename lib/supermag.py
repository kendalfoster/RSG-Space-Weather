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



################################################################################
####################### Restructure ############################################
### Function to restructure the SuperMAG data as a Dataset (xarray)
#       inputs: csv_file- SuperMAG data as csv file, downloaded from SuperMAG website
#               readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLAT- input True if the magnetic latitude column is included, default is non-inclusion of the column
#               MLT- input True if the magnetic local time column is included, default is non-inclusion of the column
#       output: Dataset with the SuperMAG data easily accessible,
#                       time is first dimension (ie, axis=0 for numpy commands)
#                       data is accessible in array format via output.measurements
#
def mag_csv_to_Dataset(csv_file, readings=None, MLT=None, MLAT=None):
    # get universally needed things
    data = pd.read_csv(csv_file)
    times = pd.to_datetime(data['Date_UTC'].unique())

    #-----------------------------------------------------------------------
    #---------- optional arguments -----------------------------------------
    #-----------------------------------------------------------------------

    # check if readings are provided
    if readings is None:
        readings = ['N', 'E', 'Z']

    # if MLAT is included, sort and make Dataset
    if MLAT is True:
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
    elif MLAT is not True:
        stations = data['IAGA'].unique()

    # if MLT (Magnetic Local Time) is included, make a Dataset
    if MLT is True:
        # initialize DataArray (so we can append things to it later)
        cols_mlt = ['Date_UTC', 'MLT']
        temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == stations[0]]
        temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
        mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                           coords = [temp_times_mlt],
                           dims = ['time'])
        # loop through the stations and append each to master DataArray
        for i in stations[1:]:
            temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == i]
            temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
            temp_mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                                    coords = [temp_times_mlt],
                                    dims = ['time'])
            mlt = xr.concat([mlt, temp_mlt], dim = 'station')
            mlt = mlt.transpose('time', 'station')
        # build MLT Dataset, for merging later
        ds_mlt = xr.Dataset(data_vars = {'mlts': (['time', 'station'], mlt)},
                            coords = {'time': times,
                                      'reading': readings,
                                      'station': stations})



    #-----------------------------------------------------------------------
    #---------- build the main DataArray of the measurements ---------------
    #-----------------------------------------------------------------------

    # initialize DataArray (so we can append things to it later)
    cols = np.append('Date_UTC', readings)
    temp_data = data[cols].loc[data['IAGA'] == stations[0]]
    temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
    da = xr.DataArray(data = temp_data[readings],
                      coords = [temp_times, readings],
                      dims = ['time', 'reading'])

    # loop through the stations and append each to master DataArray
    for i in stations[1:]:
        temp_data = data[cols].loc[data['IAGA'] == i]
        temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
        temp_da = xr.DataArray(data = temp_data[readings],
                               coords = [temp_times, readings],
                               dims = ['time', 'reading'])
        da = xr.concat([da, temp_da], dim = 'station')
        da = da.transpose('time', 'reading', 'station')

    # build Dataset from readings
    ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading', 'station'], da)},
                    coords = {'time': times,
                              'reading': readings,
                              'station': stations})



    #-----------------------------------------------------------------------
    #---------- build the final DataArray from optional arguments ----------
    #-----------------------------------------------------------------------

    # include MLT
    if MLT is True:
        ds = xr.merge([ds, ds_mlt])

    # include MLAT
    if MLAT is True:
        ds = xr.merge([ds, ds_mlat])


    return ds
################################################################################





################################################################################
####################### Plotting ###############################################
### Function to plot the readings like on the SuperMAG website
#       input: ds- dataset output from mag_data_to_Dataset function
#       output: series of plots, one per station, of the readings
def plot_mag_data(ds):
    ds.measurements.plot.line(x='time', hue='reading', col='station', col_wrap=1)
################################################################################






###############################################################################
####################### Detrending #############################################
### Function to detrend the data over time
#       input: ds- dataset output from mag_csv_to_Dataset function
#              type- type of detrending passed to scipy detrend, default linear
#       output: Dataset with measurements detrended
#                   data_vars- measurements
#                   coordinates- time, reading, station
def mag_detrend(ds, type='linear'):
    stations = ds.station
    readings = ds.reading

    # initialize DataArray
    temp = ds.measurements.loc[dict(station = stations[0])]
    temp = temp.dropna(dim = 'time', how = 'any')
    temp_times = temp.time
    det = scg.detrend(data=temp, axis=0, type=type)
    da = xr.DataArray(data = det,
                      coords = [temp_times, readings],
                      dims = ['time', 'reading'])

    for i in range(1, len(stations)):
        temp = ds.measurements.loc[dict(station = stations[i])]
        temp = temp.dropna(dim = 'time', how = 'any')
        temp_times = temp.time
        det = scg.detrend(data=temp, axis=0, type=type)
        temp_da = xr.DataArray(data = det,
                               coords = [temp_times, readings],
                               dims = ['time', 'reading'])
        da = xr.concat([da, temp_da], dim = 'station')

    # fix coordinates
    da = da.assign_coords(station = stations)

    # convert DataArray into Dataset
    res = da.to_dataset(name = 'measurements')

    return res

################################################################################






################################################################################
####################### Windowing ##############################################
### Function to window the data at a given window length
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#       output: Dataset with extra dimension from windowing
#                   data_vars- measurements, mlts, mlats
#                   coordinates- time, reading, station, window
def window(ds, win_len=128):
    # create a rolling object
    ds_roll = ds.rolling(time=win_len).construct(window_dim='win_rel_time').dropna('time')
    # fix window coordinates
    ds_roll = ds_roll.assign_coords(win_rel_time = range(win_len))
    ds_roll = ds_roll.rename(dict(time = 'win_start'))
    # ensure the coordinates are in the proper order
    ds_roll = ds_roll.transpose('win_start', 'reading', 'station', 'win_rel_time')

    return ds_roll
################################################################################




################################################################################
####################### Canonical Correlation Analysis #########################
### Function to calculate the first canonical correlation coefficients between stations
#       input: ds- dataset output from mag_data_to_Dataset function
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_st', 'second_st'
def inter_st_cca(ds, readings=None):
    # check if readings are provided
    if readings is None:
        readings = ['N', 'E', 'Z']

    # universally necessary things
    stations = ds.station
    num_st = len(stations)
    num_read = len(readings)

    # setup (triangular) array for the correlation coefficients
    cca_coeffs = np.zeros(shape = (num_st, num_st), dtype = float)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        first_st = ds.measurements.loc[dict(station = stations[i])]
        for j in range(i+1, num_st):
            second_st = ds.measurements.loc[dict(station = stations[j])]
            # remove NaNs from data (will mess up cca)
            comb_st = xr.concat([first_st, second_st], dim = 'reading')
            comb_st = comb_st.dropna(dim = 'time', how = 'any')
            first_st = comb_st[:, 0:num_read]
            second_st = comb_st[:, num_read:2*num_read]
            # run cca
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
            cca_coeffs[i,j] = temp_cca.train([first_st, second_st]).cancorrs[0]

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [stations, stations],
                      dims = ['first_st', 'second_st'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res


### Function to calculate the first canonical correlation coefficients between readings in one station
#       input: ds- dataset output from mag_data_to_Dataset function
#              station- 3 letter code for the station as a string, ex: 'BLC'
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read'
def intra_st_cca(ds, station, readings=None):
    # check if readings are provided
    if readings is None:
        readings = ['N', 'E', 'Z']

    # universally necessary things
    num_read = len(readings)

    # get readings for the station
    read = ds.measurements.loc[dict(station = station)]
    # remove Nans from data (will mess up cca)
    read = read.dropna(dim = 'time', how = 'any')

    # setup (triangular) array for the correlation coefficients
    cca_coeffs = np.zeros(shape = (num_read, num_read), dtype = float)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_read-1):
        first_read = read.loc[dict(reading = readings[i])].values
        first_read = np.reshape(first_read, newshape=[len(first_read),1])
        for j in range(i+1, num_read):
            second_read = read.loc[dict(reading = readings[j])].values
            second_read = np.reshape(second_read, newshape=[len(second_read),1])
            # run cca
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
            cca_coeffs[i,j] = abs(temp_cca.train([first_read, second_read]).cancorrs[0])

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [readings, readings],
                      dims = ['first_read', 'second_read'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res


### Function to calculate the first canonical correlation coefficients between readings for all stations
#       input: ds- dataset output from mag_data_to_Dataset function
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read', 'station'
def st_cca(ds, readings=None):
    # universally necessary things
    stations = ds.station.values
    num_st = len(stations)

    # initialize result Dataset (so we can append things to it later)
    res = intra_st_cca(ds = ds, station = stations[0], readings = readings)

    # loop through the stations and append each to master Dataset
    for i in stations[1:]:
        temp_ds = intra_st_cca(ds = ds, station = i, readings = readings)
        res = xr.concat([res, temp_ds], dim = 'station')

    # fix coordinates for 'station' dimension
    res = res.assign_coords(station = stations)

    return res

################################################################################

#### Function to remove n/a values in both time slots between two stations
#    input: ds - dataset output from mag_csv_to_Dataset Function
#           station 1 - 3 letter code for the station 1 as a string, ex: 'BLC'
#           station 2 - 3 letter code for the station 2 as a string, ex: 'BLC'
#           readings - Vector of characters representing measurements, default is ['N', 'E', 'Z']
#    output: data1 - station 1 data
#            data2 - station 2 data
#            coordinates- 'first_read', 'second_read', 'station'

def cleans_na(ds, station1, station2, readings=None):
    # check if readings are provided
    if readings is None:
        readings = ['N', 'E', 'Z']

    #Read data and merge
    read1 = ds1.measurements.loc[dict(station = station1)]
    read2 = ds1.measurements.loc[dict(station = station2)]
    merge = xr.concat([read1, read2], dim = 'station')

    #Drop n/a values
    mergenew = merge.dropna(dim = 'time', how = 'any')

    #Split apart again
    data1 = mergenew[0]
    data2 = mergenew[1]

    return (data1, data2)




#### Function to calculate the first canonical correlation coefficients between the direction
#    reading for two stations
#    input: ds - dataset output from mag_csv_to_Dataset Function
#           station 1 - 3 letter code for the station 1 as a string, ex: 'BLC'
#           station 2 - 3 letter code for the station 2 as a string, ex: 'BLC'
#           readings - Vector of characters representing measurements, default is ['N', 'E', 'Z']
#    output: data- cca_coeffs
#                   coordinates- 'first_read', 'second_read', 'station'

def inter_direction_cca(ds, station1, station2, readings=None):
     # check if readings are provided
      if readings is None:
          readings = ['N', 'E', 'Z']

      # universally necessary things
      num_read = len(readings)

      # get readings for the station
      data1, data2 = cleans_na(ds, station1, station2)

      # setup row array for the correlation coefficients
      cca_coeffs = np.zeros(shape = (1, num_read), dtype = float)

      #Calculate the cannonical correlation between the directional meaurements on each station
      for i in range(num_read):
          first_read = data1[:,i].data
          first_read = np.reshape(first_read, newshape=[len(first_read),1])

          second_read = data2[:,i].data
          second_read = np.reshape(second_read, newshape=[len(second_read),1])

          temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
          cca_coeffs[0,i] = abs(temp_cca.train([first_read, second_read]).cancorrs[0])
      return cca_coeffs


##### Function that finds the three directional correaltion coefficients at
#     two different window start times
#     input: ds - dataset output from mag_csv_to_Dataset Function
#           station 1 - 3 letter code for the station 1 as a string, ex: 'BLC'
#           station 2 - 3 letter code for the station 2 as a string, ex: 'BLC'
#           wind_start1 - Index of start of window for station1
#           wind_start2 - Index of start of window for station2
#    output: cca_coeffs - the three directional correlation coefficients



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



################################################################################
####################### Thresholding ###########################################
### Function to calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st'
def mag_thresh_kf(ds, readings=['N', 'E', 'Z']):
    thr = inter_st_cca(ds=ds, readings=readings)
    thr = thr.rename(dict(cca_coeffs = 'thresholds'))
    return thr


### Function to calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#              n0- the desired expected normalized degree of each node (station)
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st'
def mag_thresh_dods(ds, n0=0.25, readings=['N', 'E', 'Z']):
    # univeral constants
    stations = ds.station.values
    num_st = len(stations)
    ct_mat = inter_st_cca(ds=ds, readings=readings)
    ct_vec = np.linspace(start=0, stop=1, num=101)

    # initialize
    arr = np.zeros(shape = (len(ct_vec), num_st))
    # iterate through all possible ct values
    for i in range(len(ct_vec)):
        temp = ct_mat.where(ct_mat > ct_vec[i], 0) # it looks opposite, but it's right
        temp = temp.where(temp <= ct_vec[i], 1)
        for j in range(num_st):
            arr[i,j] = sum(temp.loc[dict(first_st = stations[j])].cca_coeffs.values) + sum(temp.loc[dict(second_st = stations[j])].cca_coeffs.values)
    # normalize
    arr = arr/(num_st-1)

    # find indices roughly equal to n0 and get their values
    idx = np.zeros(num_st, dtype=int)
    thr = np.zeros(num_st)
    for i in range(num_st):
        idx[i] = int(np.where(arr[:,i] <= n0)[0][0])
        thr[i] = ct_vec[idx[i]]

    # create threshold matrix using smaller threshold in each pair
    threshold = np.ones(shape = (num_st, num_st))
    for i in range(num_st):
        for j in range(i+1, num_st):
            if thr[i] < thr[j]:
                threshold[i,j] = thr[i]
                threshold[j,i] = thr[i]
            else:
                threshold[i,j] = thr[j]
                threshold[j,i] = thr[j]

    # restructure into Dataset
    res = xr.Dataset(data_vars = {'thresholds': (['first_st', 'second_st'], threshold)},
                     coords = {'first_st': stations,
                               'second_st': stations})

    return res
################################################################################




################################################################################
####################### Constructing the Network ###############################
### Function to ultimately calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#              n0- the desired expected normalized degree of each node (station)
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st', 'win_start'
def construct_network(ds, win_len=128, n0=0.25, readings=['N', 'E', 'Z']):
    # run window over data
    ds_win = sm.window(ds=ds, win_len=win_len)

    # format Dataset
    ds_win = ds_win.transpose('win_rel_time', 'reading', 'station', 'win_start')
    ds_win = ds_win.rename(dict(win_rel_time = 'time'))

    # get threshold values for each window
    det = sm.mag_detrend(ds = ds_win[dict(win_start = 0)])
    net = sm.mag_thresh_dods(ds = det, n0=n0, readings=readings)
    for i in range(1, len(ds_win.win_start)):
        det = sm.mag_detrend(ds = ds_win[dict(win_start = i)])
        temp = sm.mag_thresh_dods(ds = det, n0=n0, readings=readings)
        net = xr.concat([net, temp], dim = 'win_start')

    # fix coordinates
    net = net.assign_coords(win_start = ds_win.win_start.values)

    return net
################################################################################
