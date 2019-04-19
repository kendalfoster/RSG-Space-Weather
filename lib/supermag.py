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
####################### Restructure ############################################
### Function to restructure the SuperMAG data as an xarray.Dataset
#       inputs: csv_file- SuperMAG data as csv file, downloaded from SuperMAG website
#               readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLT- input False if the magnetic local time column is NOT included, default is True
#               MLAT- input False if the magnetic latitude column is NOT included, default is True
#       output: Dataset with the SuperMAG data easily accessible,
#                   data_vars- measurements, mlts, mlats
#                   coordinates- time, reading, station
#                 ----time is first dimension (ie, axis=0 for numpy commands)
#                 ----data in DataArray format via output.measurements
#                 ----data in array format via output.measurements.values
def mag_csv_to_Dataset(csv_file, readings=['N', 'E', 'Z'], MLT=True, MLAT=True):
    # get universally needed things
    data = pd.read_csv(csv_file)
    times = pd.to_datetime(data['Date_UTC'].unique())

    #-----------------------------------------------------------------------
    #---------- optional arguments -----------------------------------------
    #-----------------------------------------------------------------------

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
#       input: ds- dataset output from mag_csv_to_Dataset function
#       output: series of plots, one per station, of the readings
def plot_mag_data(ds):
    ds.measurements.plot.line(x='time', hue='reading', col='station', col_wrap=1)
################################################################################




################################################################################
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
#       input: ds- dataset output from mag_csv_to_Dataset function
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_st', 'second_st'
def inter_st_cca(ds, readings=['N', 'E', 'Z']):
    # universal constants
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
            # run cca, suppress rcca output
            sys.stdout = open(os.devnull, "w")
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
            cca_coeffs[i,j] = temp_cca.train([first_st, second_st]).cancorrs[0]
            sys.stdout = sys.__stdout__

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [stations, stations],
                      dims = ['first_st', 'second_st'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res


### Function to calculate the first canonical correlation coefficients between readings in one station
#       input: ds- dataset output from mag_csv_to_Dataset function
#              station- 3 letter code for the station as a string, ex: 'BLC'
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read'
def intra_st_cca(ds, station, readings=['N', 'E', 'Z']):
    # universal constants
    num_read = len(readings)

    # get readings for the station
    read = ds.measurements.loc[dict(station = station)]
    # remove Nans from data (will mess up cca)
    read = read.dropna(dim = 'time', how = 'any')

    # setup (triangular) array for the correlation coefficients
    cca_coeffs = np.zeros(shape = (num_read, num_read), dtype = float)

    # shrinking nested for loops to get all the pairs of readings
    for i in range(0, num_read-1):
        first_read = read.loc[dict(reading = readings[i])].values
        first_read = np.reshape(first_read, newshape=[len(first_read),1])
        for j in range(i+1, num_read):
            second_read = read.loc[dict(reading = readings[j])].values
            second_read = np.reshape(second_read, newshape=[len(second_read),1])
            # run cca, suppress rcca output
            sys.stdout = open(os.devnull, "w")
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
            cca_coeffs[i,j] = abs(temp_cca.train([first_read, second_read]).cancorrs[0])
            sys.stdout = sys.__stdout__

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [readings, readings],
                      dims = ['first_read', 'second_read'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res


### Function to calculate the first canonical correlation coefficients between readings for all stations
#       input: ds- dataset output from mag_csv_to_Dataset function
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read', 'station'
def st_cca(ds, readings=['N', 'E', 'Z']):
    # universal constants
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




################################################################################
####################### Phase Correlation ######################################
### Function to calculate the maximum phase correlation between two DataArrays
#       input: first_da- 1-dimensional DataArray with coords 'time' and 'win_start'
#              second_da- 1-dimensional DataArray with coords 'time' and 'win_start'
#       output: maximum correlation via windowing (float)
def max_phase_corr(first_da, second_da):
    max_corr = 0
    first_da = first_da.transpose('time', 'win_start')
    second_da = second_da.transpose('time', 'win_start')

    for i in range(len(first_da.win_start)):
        first_det = scg.detrend(data=first_da[dict(win_start = i)], axis=0)
        first_det = np.reshape(first_det, newshape=[len(first_det),1])
        for j in range(len(second_da.win_start)):
            second_det = scg.detrend(data=second_da[dict(win_start = j)], axis=0)
            second_det = np.reshape(second_det, newshape=[len(second_det),1])
            # run cca, suppress rcca output
            sys.stdout = open(os.devnull, "w")
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
            cur_corr = temp_cca.train([first_det, second_det]).cancorrs[0]
            sys.stdout = sys.__stdout__
            # check if this is bigger than max_corr
            if cur_corr > max_corr:
                max_corr = cur_corr

    return max_corr


### Function to calculate the phase correlation coefficients between readings between stations
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_st', 'second_st', 'reading'
def inter_st_phase_cca(ds, win_len=128, readings=['N', 'E', 'Z']):
    # universal constants
    stations = ds.station.values
    num_st = len(stations)
    num_read = len(readings)

    # window the Dataset
    ds_win = window(ds = ds, win_len = win_len)

    # fix coordinates
    ds_win = ds_win.rename(dict(win_rel_time = 'time'))
    ds_win = ds_win.transpose('time', 'reading', 'station', 'win_start')
    # remove Nans from data (will mess up cca)
    ds_win = ds_win.dropna(dim = 'time', how = 'any')

    # initialize array
    cca_coeffs = np.zeros(shape = (num_st, num_st, num_read))

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        first_st = ds_win.measurements.loc[dict(station = stations[i])]
        for j in range(i+1, num_st):
            second_st = ds_win.measurements.loc[dict(station = stations[j])]
            # loop through the readings
            for k in range(num_read):
                cca_coeffs[0,1,k] = max_phase_corr(first_st.loc[dict(reading = readings[k])],
                                                   second_st.loc[dict(reading = readings[k])])

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [stations, stations, readings],
                      dims = ['first_st', 'second_st', 'reading'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res



### Function to calculate the phase correlation coefficients between readings in one station
#       input: ds- dataset output from mag_csv_to_Dataset function
#              station- 3 letter code for the station as a string, ex: 'BLC'
#              win_len- length of the window, default is 128
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read'
def intra_st_phase_cca(ds, station, win_len=128, readings=['N', 'E', 'Z']):
    # universal constants
    num_read = len(readings)

    # window the Dataset
    ds_win = window(ds = ds, win_len = win_len)

    # get readings for the station
    read = ds_win.measurements.loc[dict(station = station)]
    # fix coordinates
    read = read.rename(dict(win_rel_time = 'time'))
    read = read.transpose('time', 'reading', 'win_start')
    # remove Nans from data (will mess up cca)
    read = read.dropna(dim = 'time', how = 'any')

    # setup (triangular) array for the correlation coefficients
    cca_coeffs = np.zeros(shape = (num_read, num_read), dtype = float)

    # shrinking nested for loops to get all the pairs of readings
    for i in range(0, num_read-1):
        first_read = read.loc[dict(reading = readings[i])]
        for j in range(i+1, num_read):
            second_read = read.loc[dict(reading = readings[j])]
            # run phase correlation for this pair of readings
            max_corr = 0
            for ii in range(len(first_read.win_start)):
                first_det = scg.detrend(data=first_read[dict(win_start = ii)], axis=0)
                first_det = np.reshape(first_det, newshape=[len(first_det),1])
                for jj in range(len(second_read.win_start)):
                    second_det = scg.detrend(data=second_read[dict(win_start = jj)], axis=0)
                    second_det = np.reshape(second_det, newshape=[len(second_det),1])
                    # run cca, suppress rcca output
                    sys.stdout = open(os.devnull, "w")
                    temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
                    cur_corr = temp_cca.train([first_det, second_det]).cancorrs[0]
                    sys.stdout = sys.__stdout__
                    # check if this is bigger than max_corr
                    if cur_corr > max_corr:
                        max_corr = cur_corr
            # store the maximum phase correlation in the output array
            cca_coeffs[i,j] = max_corr

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [readings, readings],
                      dims = ['first_read', 'second_read'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res


### Function to calculate the phase correlation coefficients between readings for all stations
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read', 'station'
def st_phase_cca(ds, win_len=128, readings=['N', 'E', 'Z']):
    # universal constants
    stations = ds.station.values
    num_st = len(stations)

    # initialize result Dataset (so we can append things to it later)
    res = intra_st_phase_cca(ds = ds, station = stations[0], win_len=128, readings = readings)

    # loop through the stations and append each to master Dataset
    for i in stations[1:]:
        temp_ds = intra_st_phase_cca(ds = ds, station = i, readings = readings)
        res = xr.concat([res, temp_ds], dim = 'station')

    # fix coordinates for 'station' dimension
    res = res.assign_coords(station = stations)

    return res
################################################################################




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
