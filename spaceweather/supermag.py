"""
Contents
--------

- supermag
"""


## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat


def supermag(csv_file=None, ds=None, win_len=128, lag_range=10, **kwargs):
    '''
    Takes data from the SuperMAG website, windows it, and creates a network of
    stations that are connected based on canonical correlation coefficients.

    Various kwargs may pertain to the following functions:\n
    :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`\n
    :func:`spaceweather.analysis.data_funcs.detrend`\n
    :func:`spaceweather.analysis.data_funcs.window`\n
    :func:`spaceweather.analysis.threshold.threshold`\n
    :func:`spaceweather.analysis.threshold.max_corr_lag`\n
    :func:`spaceweather.analysis.threshold.adj_mat`\n
    :func:`spaceweather.rcca`\n

    Parameters
    ----------
    csv_file : csv file
        CSV file downloaded from the SuperMAG website.
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    win_len : int, optional
        Length of window in minutes. Default is 128.
    lag_range: int, optional
        The range, in minutes, of positive and negative shifts for the second
        station in each pair. Default is 10.

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

    # get adjacency matrix using lags
    adj_mat = sat.adj_mat(ds = ds, win_len = win_len,
                          lag_range = lag_range, **kwargs)

    # plot the network, or animate a gif or something later on

    return adj_mat
