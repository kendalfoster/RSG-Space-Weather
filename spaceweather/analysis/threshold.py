## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.visualisation.heatmaps as svh


def thresh_kf(ds, **kwargs):
    """
    Calculate the threshold for each station pair, my method.

    This function simply uses the first canonical correlation coefficients
    across the entire time series for each station pair.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for each station pair.
            The data_vars are: thresholds.\n
            The coordinates are: first_st, second_st.
    """

    thr = sac.cca_coeffs(ds=ds, **kwargs)
    thr = thr.rename(dict(cca_coeffs = 'thresholds'))
    return thr


def thresh_dods(ds, n0=None, **kwargs):
    """
    Calculate the threshold for each station pair using the method in the
    Dods et al (2015) paper.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    n0 : float, optional
        The desired expected normalized degree of each station.
        Default is 1/[number of stations - 1].

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for each station pair.
            The data_vars are: thresholds.\n
            The coordinates are: first_st, second_st.
    """

    nn = kwargs.get('n0', None)
    if nn is not None:
        n0 = nn

    # univeral constants
    components = ds.component.values
    stations = ds.station.values
    num_st = len(stations)
    if n0 is None:
        n0 = 1/(num_st-1)
    cca_coeffs = sac.cca_coeffs(ds=ds, **kwargs)
    ct_vec = np.linspace(start=0, stop=1, num=101)

    # initialize
    ct_arr = np.zeros(shape = (len(ct_vec), num_st))
    # iterate through all possible ct values
    for i in range(len(ct_vec)):
        temp = cca_coeffs.where(cca_coeffs > ct_vec[i], 0) # it looks opposite, but it's right
        temp = temp.where(temp <= ct_vec[i], 1)
        for j in range(num_st):
            ct_arr[i,j] = sum(temp.loc[dict(first_st = stations[j])].cca_coeffs.values)
    # normalize
    ct_arr = ct_arr/(num_st-1)

    # find indices roughly equal to n0 and get their values
    thr = np.zeros(num_st)
    for i in range(num_st):
        thr[i] = ct_vec[int(np.where(ct_arr[:,i] <= n0)[0][0])]

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


def threshold(ds, thr_meth='Dods', **kwargs):
    """
    Calculate the pairwise thresholds for each station pair, using the provided method.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    thr_meth : str, optional
        The method used to calculate the threshold. Options are 'Dods' and 'kf'.
        Default is 'Dods'. Note you may have to add kwargs for the method.

    Returns
    -------
    xarray.Dataset
        Dataset containing the thresholds for each station pair.
            The data_vars are: thresholds.\n
            The coordinates are: first_st, second_st.
    """

    # determine method of thresholding
    if thr_meth is 'Dods':
        return thresh_dods(ds=ds, **kwargs)
    elif thr_meth is 'kf':
        return thresh_kf(ds=ds, **kwargs)
    else:
        print('Error: not a valid thresholding method')
        return 'Error: not a valid thresholding method'


def adj_mat(ds, thr_xrds=None, thr_array=None, thr_ds=None, thr_meth='Dods',
            plot=False, **kwargs):
    """
    Calculate the adjacency matrix for a set of stations.

    This function calculates the pairwise correlations between the stations in ds.
    Then it determines adjacency by comparing those correlations with either\n
        1) the thresholds provided in thr_xrds,\n
        2) the thresholds provided in thr_array, or\n
        3) the thresholds calculated from thr_ds.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
        This window of a Dataset is used to calculate the pairwise correlations,
        for comparison with the pairwise thresholds.
    thr_xrds : xarray.Dataset, optional
        xarray.Dataset containing the threshold values for ds.
        If not included, either thr_array or thr_ds must be included.
    thr_array : np.ndarray, optional
        Numpy array containing the threshold values for ds.
        If not included, either thr_xrds or thr_ds must be included.
    thr_ds : xarray.Dataset, optional
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This is used to calculate the pairwise thresholds and must contain the
        same stations as ds. Often this is a longer time series, of which ds is
        a window. If not included, either thr_xrds or thr_array must be included.
    thr_meth : str, optional
        The method used to calculate the threshold. Options are 'Dods' and 'kf'.
        Default is 'Dods'. Note you may have to add kwargs for the method.
    plot : bool, optional
        Whether or not to plot the adjacency matrix as a heatmap. Default is False.

    Returns
    -------
    xarray.Dataset
        Dataset containing the adjacency coefficients.
            The data_vars are: adj_coeffs.\n
            The coordinates are: first_st, second_st.
    """

    # check if plot is in kwargs
    pp = kwargs.get('plot', None)
    if pp is not None:
        plot = pp

    # get constants
    stations = ds.station.values
    num_st = len(stations)
    rns = range(num_st)

    if thr_xrds is None:
        if thr_array is None:
            if thr_ds is None:
                print('Error: you must include one of:\n 1) thr_xrds\n 2) thr_array\n 3) thr_ds')
                # return 'Error: you must include one of:\n 1) thr_xrds\n 2) thr_array\n 3) thr_ds'
            else:
                thresh = threshold(ds=thr_ds, thr_meth=thr_meth, **kwargs)
                thresh = thresh.assign_coords(first_st = rns, second_st = rns)
        else:
            thresh = xr.Dataset(data_vars = {'thresholds': (['first_st', 'second_station'], thr_array)},
                                coords = {'first_st': rns,
                                          'second_st': rns})
    else:
        thresh = thr_xrds.assign_coords(first_st = rns, second_st = rns)

    # calculate pairwise CCA coefficients
    cca = sac.cca_coeffs(ds=ds, **kwargs)
    cca = cca.assign_coords(first_st = rns, second_st = rns)

    adj_mat = cca - thresh.thresholds
    values = adj_mat.cca_coeffs.values
    values[values > 0] = 1
    values[values <= 0] = 0
    adj_mat.cca_coeffs.values = values
    adj_mat = adj_mat.assign_coords(first_st = stations, second_st = stations)
    adj_mat = adj_mat.rename(name_dict=dict(cca_coeffs = 'adj_coeffs'))

    # plot adjacency matrix
    if plot:
        fig = svh.plot_adj_mat(adj_mat = adj_mat, stations = stations, rns = rns)

    return adj_mat
