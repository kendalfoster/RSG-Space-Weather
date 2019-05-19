## Packages
import numpy as np
import scipy.signal as signal
import pandas as pd
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca


## Dependencies
# numpy
# scipy
# matplotlib.pyplot
# pandas
# xarray
# cartopy
# rcca (code downloaded from GitHub)


## Unused Packages, but potentially useful
# import xscale.signal.fitting as xsf # useful functions for xarray data structures
    # pip3 install git+https://github.com/serazing/xscale.git
    # pip3 install toolz



################################################################################
####################### Restructure ############################################
def csv_to_Dataset(csv_file, components=['N', 'E', 'Z'], MLT=False, MLAT=False, **kwargs):
    """
    Read the SuperMAG data as an xarray Dataset, with time as the first dimension.

    Parameters
    ----------
    csv_file : csv file
        CSV file downloaded from the SuperMAG website.
    components : list, optional
        List of components in the data. Default is ['N', 'E', 'Z'].
    MLT : bool, optional
        Whether or not MLT data is included in csv_file. Default is False.
    MLAT : bool, optional
        Whether or not MLAT data is included in csv_file. Default is False.

    Returns
    -------
    xarray.Dataset
        Dataset with the SuperMAG data easily accessible.
            The data_vars are: measurements, mlts, mlats.\n
            The coordinates are: time, component, station.

    """

    # check if kwargs contains components, MLT, or MLAT
    cc = kwargs.get('components', None)
    if cc is not None:
        components = cc

    mm = kwargs.get('MLT', None)
    if mm is not None:
        MLT = mm

    aa = kwargs.get('MLAT', None)
    if aa is not None:
        MLAT = aa

    # universal constants
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
        if num_st > 1:
            mlat_arr = np.vstack((stations,np.zeros(num_st))).transpose()
            for i in range(0,num_st):
                mlat_arr[i,1] = data['MLAT'].loc[data['IAGA'] == stations[i]].mean()
            mlat_arr = sorted(mlat_arr, key=lambda x: x[1], reverse=True)
            stations = [i[0] for i in mlat_arr]
            mlats = [round(i[1],4) for i in mlat_arr]
            # build MLAT Dataset, for merging later
            ds_mlat = xr.Dataset(data_vars = {'mlats': (['station'], mlats)},
                                 coords = {'station': stations})
        else: # if only one station
            mlats = data['MLAT'].loc[data['IAGA'] == stations[0]].mean()
            da_mlat = xr.DataArray(data = mlats)
            da_mlat = da_mlat.expand_dims(station = stations)
            # convert DataArray into Dataset, for merging later
            ds_mlat = da_mlat.to_dataset(name = 'mlats')
    else:
        stations = data['IAGA'].unique()
        num_st = len(stations)

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
        if num_st > 1:
            for i in stations[1:]:
                temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == i]
                temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
                temp_mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                                        coords = [temp_times_mlt],
                                        dims = ['time'])
                mlt = xr.concat([mlt, temp_mlt], dim = 'station')
                mlt = mlt.transpose('time', 'station')
        else: # if only one station
            mlt = mlt.expand_dims(station = stations)
        # convert DataArray into Dataset, for merging later
        ds_mlt = mlt.to_dataset(name = 'mlts')


    #-----------------------------------------------------------------------
    #---------- build the main DataArray of the measurements ---------------
    #-----------------------------------------------------------------------

    # initialize DataArray (so we can append things to it later)
    cols = np.append('Date_UTC', components)
    temp_data = data[cols].loc[data['IAGA'] == stations[0]]
    temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
    da = xr.DataArray(data = temp_data[components],
                      coords = [temp_times, components],
                      dims = ['time', 'component'])

    # loop through rest of the stations and append each to master DataArray
    if num_st > 1:
        for i in stations[1:]:
            temp_data = data[cols].loc[data['IAGA'] == i]
            temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
            temp_da = xr.DataArray(data = temp_data[components],
                                   coords = [temp_times, components],
                                   dims = ['time', 'component'])
            da = xr.concat([da, temp_da], dim = 'station')
            da = da.transpose('time', 'component', 'station')
        da = da.assign_coords(station = stations)
    else: # if only one station
        da = da.expand_dims(station = stations)

    # convert DataArray into Dataset
    ds = da.to_dataset(name = 'measurements')


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
####################### Detrending #############################################
def detrend(ds, detr_type='linear', **kwargs):
    """
    Detrend the time series for each component.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    detr_type : str, optional
        Type of detrending passed to scipy detrend. Default is 'linear'.

    Returns
    -------
    xarray.Dataset
        Dataset with the SuperMAG data easily accessible.
            The data_vars are: measurements.\n
            The coordinates are: time, component, station.
    """

    # check if kwargs contains detr_type
    d_t = kwargs.get('detr_type', None)
    if d_t is not None:
        detr_type = d_t

    stations = ds.station
    components = ds.component

    # initialize DataArray
    temp = ds.measurements.loc[dict(station = stations[0])]
    temp = temp.dropna(dim = 'time', how = 'any')
    temp_times = temp.time
    det = signal.detrend(data=temp, axis=0, type=detr_type)
    da = xr.DataArray(data = det,
                      coords = [temp_times, components],
                      dims = ['time', 'component'])

    for i in range(1, len(stations)):
        temp = ds.measurements.loc[dict(station = stations[i])]
        temp = temp.dropna(dim = 'time', how = 'any')
        temp_times = temp.time
        det = signal.detrend(data=temp, axis=0, type=detr_type)
        temp_da = xr.DataArray(data = det,
                               coords = [temp_times, components],
                               dims = ['time', 'component'])
        da = xr.concat([da, temp_da], dim = 'station')

    # fix coordinates
    da = da.assign_coords(station = stations)

    # convert DataArray into Dataset
    res = da.to_dataset(name = 'measurements')

    return res
################################################################################




################################################################################
####################### Windowing ##############################################
def window(ds, win_len=128):
    """
    Window the time series for a given window length.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    win_len : int, optional
        Length of window in minutes. Default is 128.

    Returns
    -------
    xarray.Dataset
        Dataset with the SuperMAG data easily accessible and an extra dimension from windowing.
            The data_vars are: measurements, mlts, mlats.\n
            The coordinates are: win_start, component, station, win_len.
    """

    # check for NA values in Dataset
    if True in np.isnan(ds.measurements):
        print('WARNING in spaceweather.analysis.data_funcs.window():\n Dataset contains NA values; all times with at least one NA value have been dropped.')
        ds = ds.dropna(dim = 'time', how = 'any')

    # create a rolling object
    ds_roll = ds.rolling(time=win_len).construct(window_dim='win_len').dropna(dim = 'time')
    # fix window coordinates
    ds_roll = ds_roll.assign_coords(win_len = range(win_len))
    times = ds_roll.time - np.timedelta64(win_len-1, 'm')
    ds_roll = ds_roll.assign_coords(time = times)
    ds_roll = ds_roll.rename(dict(time = 'win_start'))
    # ensure the coordinates are in the proper order
    ds_roll = ds_roll.transpose('win_start', 'component', 'station', 'win_len')

    return ds_roll
################################################################################
