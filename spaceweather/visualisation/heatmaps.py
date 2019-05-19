## Packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat


def adj_mat(ds, ds_win, n0=0.25, ret=True):
    """
    Calculate and plot the adjacency matrix for a set of stations during one time window.

    This function follows the outline in the Dods et al (2015) paper for
    calculating the pairwise thresholds. It then determines adjacency by
    comparing the correlations in the specified window of the dataset to the
    above thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This is used to calculate the pairwise thresholds.
    ds_win : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This window of a Dataset is used to calculate the pairwise correlations,
        for comparison with the above pairwise thresholds.
    n0 : float, optional
        The desired expected normalized degree of each station. Default is 0.25.
    ret : bool, optional
        Boolean value of whether or not to return related objects.
        The objects are the adjacency matrix and heatmap. Default is True.

    Returns
    -------
    xarray.Dataset
        Dataset containing the adjacency coefficients.
            The data_vars are: adj_coeffs.\n
            The coordinates are: first_st, second_st.

    matplotlib.figure.Figure
        Plot of the adjacency matrix.
    """

    stations = ds.station.values
    num_st = len(stations)
    components = ds.component.values

    cca = sac.cca_coeffs(ds=ds_win)
    cca = cca.assign_coords(first_st = range(num_st))
    cca = cca.assign_coords(second_st = range(num_st))

    thresh = sat.thresh_dods(ds=ds, n0=n0)
    thresh = thresh.assign_coords(first_st = range(num_st))
    thresh = thresh.assign_coords(second_st = range(num_st))

    adj_mat = cca - thresh.thresholds
    values = adj_mat.cca_coeffs.values
    values[values > 0] = 1
    values[values <= 0] = 0
    adj_mat.cca_coeffs.values = values
    adj_mat = adj_mat.rename(name_dict=dict(cca_coeffs = 'adj_coeffs'))

    fig = plt.figure(figsize=(10,8))
    adj_mat.adj_coeffs.plot.pcolormesh(yincrease=False, cbar_kwargs={'label': 'CCA Threshold'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title('Adjacency Matrix', fontsize=30)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(ticks=range(num_st), labels=stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(ticks=range(num_st), labels=stations, rotation=0)
    plt.show()

    if ret:
        return adj_mat, fig


def correlogram(ds, station1=None, station2=None, lag_range=10, win_len=128,
                ret=True):
    """
    Calculate and plot a heatmap of correlations between two stations over time windows and lag.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
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
    ret : bool, optional
        Boolean value of whether or not to return related objects.
        The objects are time, lag, correlations, and figure. Default is True.

    Returns
    -------
    range
        Range of times used in the correlogram.
    range
        Range of lags used in the correlogram.
    numpy.ndarray
        Numpy array of correlations.
    matplotlib.figure.Figure
        Plot of the correlogram; ie heatmap of correlations.
    """

    # check if stations are provided
    stations = ds.station.values
    if len(stations) <= 1:
        return 'Error: only one station in Dataset'
    if station1 is None:
        station1 = stations[0]
    if station2 is None:
        station2 = stations[1]

    # Select the stations and window the data
    ds = ds.loc[dict(station = [station1,station2])]
    windowed = sad.window(ds,win_len)
    ts1 = windowed.loc[dict(station = station1)].measurements
    ts2 = windowed.loc[dict(station = station2)].measurements
    ts1 = ts1.transpose('win_rel_time', 'component', 'win_start')
    ts2 = ts2.transpose('win_rel_time', 'component', 'win_start')

    # Set up array
    time = range(lag_range+1, len(windowed.win_start)-lag_range+1)
    lag = range(-lag_range, lag_range+1)
    corr = np.zeros(shape = (len(lag), len(time)))

    # Calculate correlations
    for j in range(len(time)):
        for i in range(len(lag)):
            ts1_temp = ts1[dict(win_start = time[j]-1)]
            ts2_temp = ts2[dict(win_start = time[j]+lag[i]-1)]
            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([ts1_temp, ts2_temp])
            corr[i,j] = ccac.cancorrs[0]

    # Produce heatmap
    x = np.arange(time[0], time[-1]+2)-0.5
    y = np.arange(lag[0], lag[-1]+2)-0.5
    fig = plt.figure(figsize=(10,8))
    plt.pcolormesh(x, y, corr)
    plt.title('Correlogram', fontsize=30)
    plt.xlabel('Time Window', fontsize=20)
    plt.ylabel('Lag, minutes', fontsize=20)
    # plt.xticks(x[:-1]+0.5) # show ticks on x-axis
    plt.yticks(y[:-1]+0.5)
    plt.colorbar(label='Intensity')
    fig.axes[-1].yaxis.label.set_size(20)
    plt.show()

    if ret:
        return time, lag, corr, fig
