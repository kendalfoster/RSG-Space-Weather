## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat


def supermag(csv_file=None, ds=None, thr_meth='Dods', win_len=128, **kwargs):
    '''
    Takes data from the SuperMAG website, windows it, and creates a network of
    stations that are connected based on canonical correlation coefficients.
    Various kwargs may pertain to the following functions:\n
    :func:`spaceweather.analysis.threshold.threshold`\n
    :func:`spaceweather.analysis.threshold.thresh_kf`\n
    :func:`spaceweather.analysis.threshold.thresh_dods`\n
    :func:`spaceweather.analysis.threshold.adj_mat`\n
    :func:`spaceweather.visualisation.heatmaps.plot_adj_mat`\n
    :func:`spaceweather.analysis.cca.cca_coeffs`\n
    :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`\n
    :func:`spaceweather.analysis.data_funcs.detrend`\n
    :func:`spaceweather.analysis.data_funcs.window`\n
    Parameters
    ----------
    csv_file : csv file
        CSV file downloaded from the SuperMAG website.
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    thr_meth : str, optional
        The method used to calculate the threshold. Options are 'Dods' and 'kf'.
        Default is 'Dods'. Note you may have to add kwargs for the method.
    win_len : int, optional
        Length of window in minutes. Default is 128.
    Returns
    -------
    xarray.Dataset
        Dataset containing the adjacency coefficients.
            The data_vars are: adj_coeffs.\n
            The coordinates are: first_st, second_st, win_start.
    '''

    # if data given as csv, read into Dataset
    if ds is None:
        if csv_file is None:
            print('Error: you must input data as either\n 1) csv file\n 2) xarray.Dataset')
            return 'Error: you must input data as either\n 1) csv file\n 2) xarray.Dataset'
        else:
            ds = sad.csv_to_Dataset(csv_file = csv_file, **kwargs)

    # calculate thresholds
    thresh = sat.threshold(ds, thr_meth = thr_meth, **kwargs)

    # window the data
    ds_win = sad.window(ds = ds, win_len = win_len)

    # get constants
    stations = ds.station.values
    num_st = len(stations)
    start_windows = ds_win.win_start.values
    num_win = len(start_windows)

    # initialize output Dataset and loop through each window
    adj_ds = np.zeros(shape = (num_st, num_st, num_win))
    for i in range(num_win):
        ds_temp = ds_win[dict(win_start = i)]
        ds_temp = ds_temp.rename(dict(win_len = 'time'))
        ds_temp = ds_temp.transpose('time', 'component', 'station')
        adj_ds[:,:,i] = sat.adj_mat(ds = ds_temp,
                                    thr_xrds = thresh,
                                    thr_meth = thr_meth,
                                    **kwargs).adj_coeffs.values

    # create Dataset
    res = xr.Dataset(data_vars = {'adj_coeffs': (['first_st', 'second_st', 'win_start'], adj_ds)},
                     coords = {'first_st': stations,
                               'second_st': stations,
                               'win_start': start_windows})

    # plot the network, or animate a gif or something later on

    return res
