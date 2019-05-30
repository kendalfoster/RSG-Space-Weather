"""
Contents
--------

- cca
- cca_angles
- lag_mat_pair
- lag_mat
"""


## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.analysis.data_funcs as sad
import spaceweather.visualisation.heatmaps as svh




def cca(X, Y, weights=True):
    """
    Run canonical correlation analysis between two inputs.

    Put more info in here about the method outlined in the paper.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix with some dimensions, finish later
    Y : numpy.ndarray
        Matrix with some dimensions, finish later
    weights : bool, optional
        Whether or not to return weights. Default is True.

    Returns
    -------
    float
        First canonical correlation coefficient between X and Y
    numpy.ndarray
        Weight vector for X
    numpy.ndarray
        Weight vector for Y
    """

    # center X and Y
    meanX = X.mean(axis=0)
    meanY = Y.mean(axis=0)
    X = X-meanX[np.newaxis,:]
    Y = Y-meanY[np.newaxis,:]

    # apply QR decomposition
    Qx, Rx = np.linalg.qr(X)
    Qy, Ry = np.linalg.qr(Y)

    # apply singular value decomposition
    svdInput = np.dot(Qx.T,Qy)
    U, s, Vt = np.linalg.svd(svdInput)

    # return coeff, a, b
    coeff = s[0]
    a = np.dot(np.linalg.inv(Rx), U[:,0])
    b = np.dot(np.linalg.inv(Ry.T), U[:,0])

    return coeff, a, b


def cca_angles(ds, **kwargs):
    """
    Calculate the first canonical correlation coefficients between stations and
    their associated angles:\n
    1) one weight relative to the other,\n
    2) each weight relative to its input's data

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.

    Returns
    -------
    xarray.Dataset
        Dataset containing the first canonical correlation coefficients.
            The data_vars are: cca_coeffs, ang_weight, ang_data.\n
            The coordinates are: first_st, second_st.
    """

    # detrend input Dataset, remove NAs
    ds = sad.detrend(ds, **kwargs)

    # universal constants
    stations = ds.station.values
    num_st = len(stations)
    components = ds.component.values
    num_comp = len(components)
    times = ds.time.values
    num_time = len(times)

    # setup arrays
    cca_coeffs = np.zeros(shape = (num_st, num_st), dtype = float)
    ang_weight = np.zeros(shape = (num_st, num_st), dtype = float)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        first_st = ds.loc[dict(station = stations[i])].measurements
        for j in range(i+1, num_st):
            st2 = ds.loc[dict(station = stations[j])].measurements
            # remove NaNs from data (will mess up cca)
            both_st = xr.concat([first_st, st2], dim = 'component')
            both_st = both_st.dropna(dim = 'time', how = 'any')
            temp_times = both_st.time.values
            num_tt = len(temp_times)
            st1 = both_st[:, 0:num_comp]
            st2 = both_st[:, num_comp:2*num_comp]
            ## run cca
            coeff, a, b = cca(st1.values, st2.values, weights = True)
            cca_coeffs[i,j] = cca_coeffs[j,i] = coeff
            # angles: one weight relative to the other
            wt_norm = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
            ang_weight[i,j] = np.rad2deg(np.arccos(np.clip(np.dot(a, b)/wt_norm, -1.0, 1.0)))
            # # angles: one weight relative to its input's data
            # ang_data_a_ss = np.zeros(num_tt)
            # ang_data_b_ss = np.zeros(num_tt)
            # for k in range(num_tt):
            #     xdata = st_1[dict(time=k)].values
            #     ydata = st_2[dict(time=k)].values
            #     wt_nrm0 = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(xdata**2))
            #     wt_nrm1 = np.sqrt(np.sum(b**2)) * np.sqrt(np.sum(ydata**2))
            #     ang_data_a_ss[k] = np.rad2deg(np.arccos(np.clip(np.dot(a, xdata)/wt_nrm0, -1.0, 1.0)))
            #     ang_data_b_ss[k] = np.rad2deg(np.arccos(np.clip(np.dot(b, ydata)/wt_nrm1, -1.0, 1.0)))

            # append ang_data to master Dataset: dimension = second_st
            # if j == i+1:
            #     ang_data_a_fs = ang_data_a_ss
            #     ang_data_b_fs = ang_data_b_ss
            # else:
            #     ang_data_a_fs = xr.concat([ang_data_a_fs, ang_data_a_ss], dim = 'second_st')
            #     ang_data_b_fs = xr.concat([ang_data_b_fs, ang_data_b_ss], dim = 'second_st')

        # # append to master Dataset: dimension = first_st
        # if i == 0:
        #     ang_data_a = ang_data_a_fs
        # elif i < num_st-2:
        #     ang_data = xr.concat([ang_data, ang_data_fs], dim = 'first_st')
        # else: # if i = num_st-2
        #     dummy = ang_data_fs.copy(deep = True)
        #     num_lag = len(lag_mat_ss.lag.values)
        #     num_win = len(lag_mat_ss.time_win.values)
        #     dummy.lag_coeffs.values = np.full(shape = (num_lag, num_win),
        #                                       fill_value = np.nan)
        #     dummy = dummy.assign_coords(first_st = stations[num_st-2],
        #                                 second_st = stations[num_st-2])
        #     lag_dum = xr.concat([lag_mat_ss, dummy], dim = 'second_st')
        #     lag_mat = xr.concat([lag_mat, lag_dum], dim = 'first_st')

    # build Dataset from the cca_coeffs array
    # ds = xr.Dataset(data_vars = {'cca_coeffs': (['first_st', 'second_st'], cca_coeffs),
    #                              'ang_weight': (['first_st', 'second_st'], ang_weight),
    #                              'ang_data': (['first_st', 'second_st', 'ab', 'time'], ang_data)},
    #                 coords = {'first_st': stations,
    #                           'second_st': stations,
    #                           'ab': ['a', 'b'],
    #                           'time': times})
    ds = xr.Dataset(data_vars = {'cca_coeffs': (['first_st', 'second_st'], cca_coeffs),
                                 'ang_weight': (['first_st', 'second_st'], ang_weight)},
                    coords = {'first_st': stations,
                              'second_st': stations})

    return ds


def lag_mat_pair(ds, station1=None, station2=None, lag_range=10, win_len=128,
                 plot=False, **kwargs):
    """
    Calculate and plot a heatmap of correlations between two stations over time windows and lag.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.\n
        Note the time dimension must be at least win_len + 2*lag_range.
    station1 : str, optional
        Station for which the time window is fixed.
        If no station is provided, this will default to the first station in ds.
    station2 : str, optional
        Station for which the time window will be shifted according to lag_range.
        If no station is provided, this will default to the second station in ds.
    lag_range: int, optional
        The range, in minutes, of positive and negative shifts for station2.
        Default is 10.
    win_len : int, optional
        Length of window in minutes. Default is 128.
    plot : bool, optional
        Whether or not to plot and return the correlation matrix as a heatmap.
        Default is False.

    Returns
    -------
    xarray.Dataset
        Dataset containing the correlation coefficients.
            The data_vars are: cor_coeffs.\n
            The coordinates are: time_win, lag, first_st, second_st, win_start.
    matplotlib.figure.Figure
        Plot of the correlogram; ie heatmap of correlations.
    """

    # check if ds timeseries is long enough
    nt = len(ds.time.values)
    if nt < win_len + 2*lag_range:
        print('Error: ds timeseries < win_len + 2*lag_range')
        return 'Error: ds timeseries < win_len + 2*lag_range'

    # check if stations are provided
    stations = ds.station.values
    if len(stations) <= 1:
        print('Error: only one station in Dataset')
        return 'Error: only one station in Dataset'
    if station1 is None:
        print('No station1 provided; using station1 = %s' % (stations[0]))
        station1 = stations[0]
        if station2 is None:
            print('No station2 provided; using station2 = %s' % (stations[1]))
            station2 = stations[1]
    elif station2 is None and not station1 == stations[0]:
        print('No station2 provided; using station2 = %s' % (stations[0]))
        station2 = stations[0]
    elif station2 is None and station1 == stations[0]:
        print('No station2 provided; using station2 = %s' % (stations[1]))
        station2 = stations[1]

    # Select the stations and window the data
    ds = ds.loc[dict(station = [station1, station2])]
    windowed = sad.window(ds, win_len)

    ts1 = windowed.loc[dict(station = station1)].measurements
    ts2 = windowed.loc[dict(station = station2)].measurements
    ts1 = ts1.transpose('win_len', 'component', 'win_start')
    ts2 = ts2.transpose('win_len', 'component', 'win_start')

    # Set up array
    time = range(lag_range+1, len(windowed.win_start)-lag_range+1)
    lag = range(-lag_range, lag_range+1)
    corr = np.zeros(shape = (len(lag), len(time)))

    # Calculate correlations
    for j in range(len(time)):
        for i in range(len(lag)):
            ts1_temp = ts1[dict(win_start = time[j]-1)]
            ts2_temp = ts2[dict(win_start = time[j]+lag[i]-1)]
            # run cca
            coeff = cca(ts1_temp.values, ts2_temp.values, weights = False)
            corr[i,j] = coeff

    lag_mat = xr.Dataset(data_vars = {'lag_coeffs': (['lag', 'time_win'], corr)},
                         coords = {'lag': lag,
                                   'time_win': time})

    # add first_st and second_st as coordinates
    lag_mat = lag_mat.assign_coords(first_st = station1,
                                    second_st = station2,
                                    win_start = windowed.win_start.values[time])

    # plot correlogram heatmap
    if plot:
        fig = svh.plot_lag_mat_pair(lag_mat = lag_mat, time_win = time, lag = lag)
        return lag_mat, fig
    else:
        return lag_mat


def lag_mat(ds, lag_range=10, win_len=128, **kwargs):
    """
    Calculate a heatmap of correlations between every station pair over time windows and lag.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.\n
        Note the time dimension must be at least win_len + 2*lag_range.
    lag_range: int, optional
        The range, in minutes, of positive and negative shifts for station2.
        Default is 10.
    win_len : int, optional
        Length of window in minutes. Default is 128.

    Returns
    -------
    xarray.Dataset
        Dataset containing the correlation coefficients.
            The data_vars are: cor_coeffs.\n
            The coordinates are: time_win, lag, first_st, second_st, win_start.
    """

    # get constants
    stations = ds.station.values
    num_st = len(stations)

    # check there are at least two stations
    if num_st <= 1:
        print('Error: only one station in Dataset')
        return 'Error: only one station in Dataset'

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        for j in range(i+1, num_st):
            lm = lag_mat_pair(ds = ds,
                              station1 = stations[i],
                              station2 = stations[j],
                              lag_range = lag_range,
                              win_len = win_len,
                              **kwargs)

            # append to master Dataset: dimension = second_st
            if j == i+1:
                lag_mat_ss = lm
            else:
                lag_mat_ss = xr.concat([lag_mat_ss, lm], dim = 'second_st')

        # append to master Dataset: dimension = first_st
        if i == 0:
            lag_mat = lag_mat_ss
        elif i < num_st-2:
            lag_mat = xr.concat([lag_mat, lag_mat_ss], dim = 'first_st')
        else: # if i = num_st-2
            dummy = lag_mat_ss.copy(deep = True)
            num_lag = len(lag_mat_ss.lag.values)
            num_win = len(lag_mat_ss.time_win.values)
            dummy.lag_coeffs.values = np.full(shape = (num_lag, num_win),
                                              fill_value = np.nan)
            dummy = dummy.assign_coords(first_st = stations[num_st-2],
                                        second_st = stations[num_st-2])
            lag_dum = xr.concat([lag_mat_ss, dummy], dim = 'second_st')
            lag_mat = xr.concat([lag_mat, lag_dum], dim = 'first_st')

    # finish the DataArrays
    time_wins = lag_mat_ss.time_win.values
    num_win = len(time_wins)
    lags = lag_mat_ss.lag.values
    num_lag = len(lags)
    win_sts = lag_mat_ss.win_start.values
    num_win_st = len(win_sts)
    lag_blank = np.full(shape=(num_win, num_lag, num_st, num_st, num_win_st), fill_value = np.nan)
    lag_bda = xr.Dataset(data_vars = {'lag_coeffs': (['time_win', 'lag', 'first_st', 'second_st', 'win_start'], lag_blank)},
                         coords = {'time_win': time_wins,
                                   'lag': lags,
                                   'first_st': stations,
                                   'second_st': stations,
                                   'win_start': win_sts})
    lag_mat = lag_mat.combine_first(lag_bda)

    # reorder the stations coordinates because xr.concat messed them up
    ds1 = lag_mat.loc[dict(first_st = stations[0])]
    for i in range(1, num_st):
        ds1 = xr.concat([ds1, lag_mat.loc[dict(first_st = stations[i])]], dim = 'first_st')
    ds2 = ds1.loc[dict(second_st = stations[0])]
    for i in range(1, num_st):
        ds2 = xr.concat([ds2, ds1.loc[dict(second_st = stations[i])]], dim = 'second_st')
    ds = ds2.transpose('time_win', 'lag', 'first_st', 'second_st', 'win_start')

    return ds
