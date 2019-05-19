## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad


def thresh_kf(ds):
    """
    Calculate the threshold for each station pair, my method.

    This function simply uses the first canonical correlation coefficients
    across the entire time series for each station pair.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for each station pair.
            The data_vars are: thresholds.\n
            The coordinates are: first_st, second_st.
    """

    thr = sac.cca_coeffs(ds=ds)
    thr = thr.rename(dict(cca_coeffs = 'thresholds'))
    return thr


def thresh_dods(ds, n0=0.25):
    """
    Calculate the threshold for each station pair, method used in Dods et al (2015) paper.

    This function follows the outline in the Dods et al (2015) paper.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    n0 : float, optional
        The desired expected normalized degree of each station. Default is 0.25.

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for each station pair.
            The data_vars are: thresholds.\n
            The coordinates are: first_st, second_st.
    """

    # univeral constants
    components = ds.component.values
    stations = ds.station.values
    num_st = len(stations)
    ct_mat = sac.cca_coeffs(ds=ds)
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
    threshold = np.zeros(shape = (num_st, num_st))
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


def threshold_ds(ds, win_len=128, n0=0.25):
    """
    Calculate the threshold for each station pair, using a windowed approach.

    This function windows the Dataset and then follows the outline in the
    Dods et al (2015) paper for calculating the pairwise thresholds in each window.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    win_len : int, optional
        Length of window in minutes. Default is 128.
    n0 : float, optional
        The desired expected normalized degree of each station. Default is 0.25.

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for each station pair.
            The data_vars are: thresholds.\n
            The coordinates are: first_st, second_st, win_start.
    """

    # run window over data
    ds_win = sad.window(ds=ds, win_len=win_len)

    # format Dataset
    ds_win = ds_win.transpose('win_rel_time', 'component', 'station', 'win_start')
    ds_win = ds_win.rename(dict(win_rel_time = 'time'))

    # get threshold values for each window
    det = sad.detrend(ds = ds_win[dict(win_start = 0)])
    net = thresh_dods(ds = det, n0=n0)
    for i in range(1, len(ds_win.win_start)):
        det = sad.detrend(ds = ds_win[dict(win_start = i)])
        temp = thresh_dods(ds = det, n0=n0)
        net = xr.concat([net, temp], dim = 'win_start')

    # fix coordinates
    net = net.assign_coords(win_start = ds_win.win_start.values)

    return net
