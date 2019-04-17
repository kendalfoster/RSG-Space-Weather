## Packages
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import lib.rcca as rcca



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
####################### Canonical Correlation Analysis #########################
### Function to calculate the first canonical correlation coefficients between stations
#       input: ds- dataset output from mag_csv_to_Dataset function
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_st', 'second_st'
def inter_st_cca(ds, readings=['N', 'E', 'Z']):

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
#       input: ds- dataset output from mag_csv_to_Dataset function
#              station- 3 letter code for the station as a string, ex: 'BLC'
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read'
def intra_st_cca(ds, station, readings=['N', 'E', 'Z']):

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
#       input: ds- dataset output from mag_csv_to_Dataset function
#              readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#       output: Dataset of cca coefficients
#                   data- cca_coeffs
#                   coordinates- 'first_read', 'second_read', 'station'
def st_cca(ds, readings=['N', 'E', 'Z']):
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




################################################################################
####################### Thresholding ###########################################
### Function to calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st'
def mag_thresh_kf(ds):
    thr = inter_st_cca(ds=ds)
    thr = thr.rename(dict(cca_coeffs = 'thresholds'))
    return thr


### Function to calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#              n0- the desired expected degree of each node (station)
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st'
def mag_thresh_dods(ds, n0=0.25):
    # univeral constants
    stations = ds.station.values
    num_st = len(stations)
    ct_mat = inter_st_cca(ds=ds)
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
####################### Windowing ##############################################
### Function to window the data at a given window length
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#       output: Dataset with extra dimension from windowing
#                   data_vars- measurements, mlts, mlats
#                   coordinates- time, reading, station, window
def window(ds, win_len=128):
    # create a rolling object
    ds_roll = ds.rolling(time=win_len).construct(window_dim='window').dropna('time')
    # fix window coordinates
    ds_roll = ds_roll.assign_coords(window = range(win_len))
    return ds_roll
################################################################################
