## Packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
# import spaceweather.analysis.threshold as sat


def plot_adj_mat(adj_mat, stations, rns):
    """
    Plot the adjacency matrix as a heatmap for a set of stations.

    This function is called from :func:`spaceweather.analysis.threshold.adj_mat`,
    and is not intended for external use.

    Parameters
    ----------
    adj_mat : xarray.Dataset
        The adjacency matrix to be plotted.
    stations : numpy.ndarray
        Numpy array of three-letter station codes.
    rns : range
        A range of the length of stations, explicitly rns = range(len(stations)).

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the adjacency matrix.
    """

    # relabel the coordinates so it will plot properly
    adj_mat = adj_mat.assign_coords(first_st = rns, second_st = rns)

    fig = plt.figure(figsize=(10,8))
    adj_mat.adj_coeffs.plot.pcolormesh(yincrease=False, cbar_kwargs={'label': 'CCA Threshold'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title('Adjacency Matrix', fontsize=30)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(rns, stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(rns, stations, rotation=0)
    plt.show()


def correlogram(ds, station1=None, station2=None, lag_range=10, win_len=128,
                ret=True):
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
        station1 = stations[0]
    if station2 is None:
        station2 = stations[1]

    # Select the stations and window the data
    ds = ds.loc[dict(station = [station1,station2])]
    windowed = sad.window(ds,win_len)
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
    if len(x) < 5:
        plt.xticks(x[:-1]+0.5) # show ticks on x-axis
    plt.yticks(y[:-1]+0.5)
    plt.colorbar(label='Correlation')
    fig.axes[-1].yaxis.label.set_size(20)
    plt.show()

    if ret:
        return time, lag, corr, fig
