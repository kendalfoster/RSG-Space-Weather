## Packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad



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


def plot_lag_mat(lag_mat, time_win, lag):
    """
    Plot a heatmap of correlations between two stations over time windows and lag.

    This function is called from :func:`spaceweather.analysis.cca.lag_mat`,
    and is not intended for external use.

    Parameters
    ----------
    lag_mat : numpy.ndarray
        Numpy array of correlations to be plotted.
    time_win : range
        A range of the indices of the time windows used in calculating the correlations.
    lag : range
        A range of the lags used in calculating the correlations.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the correlogram; ie heatmap of correlations.
    """

    # Produce heatmap
    x = np.arange(time_win[0], time_win[-1]+1)-0.5
    y = np.arange(lag[0], lag[-1]+1)-0.5

    fig = plt.figure(figsize=(10,8))
    lag_mat.lag_coeffs.plot.pcolormesh(cbar_kwargs={'label': 'Correlation'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title('Lag Correlogram', fontsize=30)
    plt.xlabel('Time Window', fontsize=20)
    plt.ylabel('Lag, minutes', fontsize=20)
    if len(x) < 5:
        plt.xticks(x[:-1]+0.5) # show ticks on x-axis
    plt.yticks(y[:-1]+0.5)
    plt.show()

    return fig
