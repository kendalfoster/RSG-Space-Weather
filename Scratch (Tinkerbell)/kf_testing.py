scrap way we do adj_mat

for each station pair,
    for each time window,
        get the lag which produces the maximum correlation (like from correlogram stuff)
        store all of this in a 4-dimensional Dataset
        run threshold (like avg over time series) over this to get new-style adj_mat
return the adj_mat with dimensions:
                                   - first_st
                                   - second_st
                                   - time_win (index/win_start_time, ideally win_start)

import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import numpy as np
import spaceweather.rcca as rcca
import xarray as xr # if gives error, just rerun



ds = sad.csv_to_Dataset(csv_file="Data/20190403-00-22-supermag.csv", MLAT=True)
ds = ds[dict(station = range(2))]
ds = ds[dict(time = slice(137), station = range(4))]
lags = np.array([-2,0,1,3])

thresh_lag(ds, lags)
max_corr_lag(ds, 10)

def thresh_lag(ds, lags, **kwargs):
    """
    Calculate the pairwise thresholds for one station pair, using lag.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    lags : np.ndarray
        Numpy array of lags at which maximum correlation is achieved.

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for the station pair and the lag.
            The data_vars are: thresholds.\n
            The coordinates are: lag.
    """

    # detrend the data
    ds = sad.detrend(ds, **kwargs)

    # get constants
    stations = ds.station.values
    num_st = len(stations)
    num_comp = len(ds.component.values)
    num_lag = len(lags)
    lag_range = np.max(abs(lags))
    time_fix = np.arange(lag_range, len(ds.time)-lag_range)
    ds_fix = ds[dict(time = time_fix)]

    # set up array
    thresh = np.zeros(num_lag)

    ts1 = ds_fix[dict(station = 0)].measurements
    ts2 = ds[dict(station = 1)]
    # loop through lags
    for k in range(num_lag):
        ts2_temp = ts2[dict(time = time_fix+lags[k])].measurements
        # remove NaNs from data (will mess up cca)
        both_ts = xr.concat([ts1, ts2_temp], dim = 'component')
        both_ts = both_ts.dropna(dim = 'time', how = 'any')
        ts1 = both_ts[:, 0:num_comp]
        ts2_temp = both_ts[:, num_comp:2*num_comp]
        # run cca, suppress rcca output
        temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
        ccac = temp_cca.train([ts1, ts2_temp])
        thresh[k] = ccac.cancorrs[0]

    # construct Dataset from array
    res = xr.Dataset(data_vars = {'thresholds': (['lag'], thresh)},
                     coords = {'lag': lags})

    return res


def max_corr_lag(ds, lag_range, **kwargs):
    """
    Calculate the maximum correlaion between the two stations over time.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.\n
        ds is assumed to only have two stations.
    lag_range: int
        The range, in minutes, of positive and negative shifts for station2.

    Returns
    -------
    xarray.DataArray
        DataArray containing the correlation coefficient.
            The coordinate is: lag.
    """

    # check if stations are provided
    stations = ds.station.values
    if len(stations) <= 1:
        print('Error: only one station in Dataset')
        return 'Error: only one station in Dataset'

    # get constants
    num_comp = len(ds.component.values)
    lags = np.arange(-lag_range, lag_range+1)
    num_lag = len(lags)
    time_fix = np.arange(lag_range, len(ds.time)-lag_range)
    ds_fix = ds[dict(time = time_fix)]

    # set up array
    cca_coeffs = np.zeros(num_lag)

    # loop through lags
    ts1 = ds_fix[dict(station = 0)].measurements
    for k in range(num_lag):
        ts2 = ds[dict(time = time_fix+lags[k], station = 1)].measurements
        # remove NaNs from data (will mess up cca)
        both_ts = xr.concat([ts1, ts2], dim = 'component')
        both_ts = both_ts.dropna(dim = 'time', how = 'any')
        ts1 = both_ts[:, 0:num_comp]
        ts2 = both_ts[:, num_comp:2*num_comp]
        # run cca, suppress rcca output
        temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
        ccac = temp_cca.train([ts1, ts2])
        cca_coeffs[k] = ccac.cancorrs[0]

    # pick maximum correlation
    max = np.max(cca_coeffs)
    lag = np.argmax(cca_coeffs)

    # convert to DataArray
    res = xr.DataArray(data = max)
    res = res.assign_coords(lag = lag)

    return res


def lag_adj_mat(ds, win_len=128, lag_range=10, **kwargs):
    """
    Calculate the adjacency matrix for a set of stations using lag.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.\n
        ds is assumed to only have two stations.
    win_len : int, optional
        Length of window in minutes. Default is 128.
    lag_range: int, optional
        The range, in minutes, of positive and negative shifts for station2.
        Default is 10.

    Returns
    -------
    xarray.Dataset
        Dataset containing the adjacency coefficients.
            The data_vars are: adj_coeffs.\n
            The coordinates are: first_st, second_st, win_start.
    """

    # check if ds timeseries is long enough
    nt = len(ds.time.values)
    if nt < win_len + 2*lag_range:
        print('Error: ds timeseries < win_len + 2*lag_range')
        return 'Error: ds timeseries < win_len + 2*lag_range'

    # check if stations are provided
    stations = ds.station.values
    num_st = len(stations)
    if num_st <= 1:
        print('Error: only one station in Dataset')
        return 'Error: only one station in Dataset'

    # window the data
    ds_win = sad.window(ds, win_len)

    # constants
    win_start = ds_win.win_start
    num_win = len(win_start)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        for j in range(i+1, num_st):
            for k in range(num_win):
                # calculate maximum correlation and associated lag
                temp_ds = ds_win[dict(station = [i,j], win_start = k)]
                temp_ds = temp_ds.rename(win_len = 'time')
                temp_ds = temp_ds.transpose('time', 'station', 'component')
                max_temp = max_corr_lag(ds = temp_ds, lag_range = lag_range)

                # append to master DataArray
                if k == 0:
                    max_corr = max_temp
                else:
                    max_corr = xr.concat([max_corr, max_temp], dim = 'win_start')
            max_corr = max_corr.assign_coords(win_start = win_start)

            # set up thresholds
            thresh = thresh_lag(ds = ds[dict(station = [i,j])],
                                lags = np.unique(max_corr.lag))

            # apply threshold for each time window
            for k in range(num_win):
                max_corr.values[k] = max_corr.values[k] - thresh.loc[dict(lag = max_corr[k].lag.values)].thresholds.values
            values = max_corr.values
            values[values > 0] = 1
            values[values <= 0] = 0
            max_corr.values = values

            # append to master Dataset: dimension = second_st
            if j == i+1:
                adj_mat_ss = max_corr
            else:
                adj_mat_ss = xr.concat([adj_mat_ss, max_corr], dim = 'second_st')

        # adjust second_st coordinates
        if i == num_st-1 and j == num_st:
            adj_mat_ss = adj_mat_ss.expand_dims(name = 'second_st')
        adj_mat_ss = adj_mat_ss.assign_coords(second_st = stations[range(i+1, num_st)])

        # append to master Dataset: dimension = first_st
        if i == 0:
            adj_mat = adj_mat_ss
        elif i < num_st-2:
            adj_mat = xr.concat([adj_mat, adj_mat_ss], dim = 'first_st')
        else:
            dum = np.zeros(num_win)
            dummy = xr.DataArray(data = dum,
                                 coords = [win_start],
                                 dims = ['win_start'])
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            adj_dum = xr.concat([adj_mat_ss, dummy], dim = 'second_st')
            adj_mat = xr.concat([adj_mat, adj_dum], dim = 'first_st')
            adj_mat = adj_mat.transpose('first_st', 'second_st')

    # adjust first_st coordinates
    adj_mat = adj_mat.assign_coords(first_st = stations)
