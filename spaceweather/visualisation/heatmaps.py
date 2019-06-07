"""
Contents
--------

- plot_adj_mat
- plot_lag_mat_pair
- plot_lag_mat_time
- plot_corr_thresh
- plot_cca_ang
"""


## Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as plc
from matplotlib.colors import ListedColormap




def plot_adj_mat(adj_mat):
    """
    Plot the adjacency matrix as a heatmap for a set of stations at one time.

    Parameters
    ----------
    adj_mat : xarray.Dataset
        The adjacency matrix to be plotted.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the adjacency matrix.
    """

    # constants
    stations = adj_mat.first_st.values
    rns = range(len(stations))

    # make title of heatmap
    time = pd.to_datetime(adj_mat.win_start.values)
    time_stamp = time.strftime('%Y.%m.%d %H:%M')
    title = 'Adjacency Matrix at ' + time_stamp

    # relabel the coordinates so it will plot properly
    adj_mat = adj_mat.assign_coords(first_st = rns, second_st = rns)

    fig = plt.figure(figsize=(10,8))
    adj_mat.adj_coeffs.plot.pcolormesh(yincrease = False,
                                       cmap = 'Greys',
                                       norm = plc.Normalize(0,1),
                                       cbar_kwargs = {'label': 'Adjacency'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title(title, fontsize=24)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(rns, stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(rns, stations, rotation=0)

    return fig


def plot_lag_mat_pair(lag_mat_pair):
    """
    Plot a heatmap of correlations between two stations over time windows and lag.

    This function is called from :func:`spaceweather.analysis.cca.lag_mat_pair`.

    Parameters
    ----------
    lag_mat_pair : xarray.Dataset
        The correlations to be plotted.
    time_win : range
        A range of the indices of the time windows used in calculating the correlations.
    lag : range
        A range of the lags used in calculating the correlations.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the correlogram; ie heatmap of correlations.
    """

    time_win = lag_mat_pair.time_win.values
    lag = lag_mat_pair.lag.values

    # Produce heatmap
    x = np.arange(time_win[0], time_win[-1]+1)-0.5
    y = np.arange(lag[0], lag[-1]+1)-0.5

    # make title of heatmap
    time = pd.to_datetime(lag_mat_pair.win_start.values[0])
    time_stamp = time.strftime('%Y.%m.%d %H:%M')
    title = np.asscalar(lag_mat_pair.first_st.values) + ' & ' + np.asscalar(lag_mat_pair.second_st.values) + ' at ' + time_stamp

    # Produce heatmap
    x = np.arange(time_win[0], time_win[-1]+1)-0.5
    y = np.arange(lag[0], lag[-1]+1)-0.5

    fig = plt.figure(figsize=(10,8))
    lag_mat_pair.lag_coeffs.plot.pcolormesh(cbar_kwargs={'label': 'Correlation'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title(title, fontsize=24)
    plt.xlabel('Time Window', fontsize=20)
    plt.ylabel('Lag, minutes', fontsize=20)
    if len(x) < 5:
        plt.xticks(x[:-1]+0.5) # show ticks on x-axis
    plt.yticks(y[:-1]+0.5)

    return fig


def plot_lag_mat_time(lag_mat):
    """
    Plot a heatmap of correlations between each pair of stations over time windows.

    Parameters
    ----------
    lag_mat : xarray.Dataset
        The correlations to be plotted.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the correlogram; ie heatmap of correlations.
    """

    # relabel the coordinates so it will plot properly
    stations = lag_mat.first_st.values
    rns = range(len(stations))
    lm = lag_mat.assign_coords(first_st = rns, second_st = rns)

    # make title of heatmap
    time = pd.to_datetime(lm.win_start.values)
    time_stamp = time.strftime('%Y.%m.%d %H:%M')
    title = 'Correlation at ' + time_stamp

    # plot heatmap
    fig = plt.figure(figsize=(10,8))
    lm.lag_coeffs.plot.pcolormesh(yincrease=False, cbar_kwargs={'label': 'Correlation'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title(title, fontsize=24)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(rns, stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(rns, stations, rotation=0)

    return fig


def plot_corr_thresh(corr_lag_mat):
    """
    Plot a heatmap of the threshold subtracted from the correlations between
    each station pair for one time.

    Parameters
    ----------
    corr_lag_mat : xarray.Dataset
        The values to be plotted; coordinates are 'first_st' and 'second_st'.

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap of the values.
    """

    # constants
    stations = corr_lag_mat.first_st.values
    num_st = len(stations)
    time = pd.to_datetime(corr_lag_mat.win_start.values)
    timestamp = time.strftime('%Y.%m.%d %H:%M')

    # adjust coordinates for plotting
    corr_lag_mat = corr_lag_mat.assign_coords(first_st = range(num_st),
                                              second_st = range(num_st))

    # define new colormap
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmap = ListedColormap(newcolors, name='OrangeBlue')
    norm = plc.Normalize(-0.5,0.5)

    # must run all following code simultaneously
    fig = plt.figure(figsize=(10,8))
    g = corr_lag_mat.corr_thresh.plot.pcolormesh(yincrease=False,
                                            cmap=newcmap,
                                            norm=norm,
                                            cbar_kwargs={'label': 'Correlation Coefficient - Threshold'})
    plt.title('Correlation Heatmap at %s' %timestamp, fontsize = 24)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(ticks=range(num_st), labels=stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(ticks=range(num_st), labels=stations, rotation=0)
    g.figure.axes[-1].yaxis.label.set_size(20)

    return fig


def plot_cca_ang(cca_ang, a_b):
    """
    Plot a heatmap of the CCA angles for each station pair for one time.

    Parameters
    ----------
    corr_lag_mat : xarray.Dataset
        The values to be plotted; coordinates are 'first_st' and 'second_st'.
    a_b : {'a', 'b'}
        Plot angles for weight 'a' or weight 'b'.

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap of the values.
    """

    # take weight a or b
    cca_ang = cca_ang.loc[dict(a_b = a_b)]

    # relabel the coordinates so it will plot properly
    stations = cca_ang.first_st.values
    rns = range(len(stations))
    cc = cca_ang.assign_coords(first_st = rns, second_st = rns)

    # make title of heatmap
    time = pd.to_datetime(cca_ang.time.values)
    time_stamp = time.strftime('%Y.%m.%d %H:%M')
    title = 'CCA Angles at ' + time_stamp

    # plot heatmap
    fig = plt.figure(figsize=(10,8))
    cc.ang_data.plot.pcolormesh(yincrease = False,
                                cmap = 'viridis',
                                norm = plc.Normalize(0,180),
                                cbar_kwargs={'label': 'Angle, degrees'})
    fig.axes[-1].yaxis.label.set_size(20)
    plt.title(title, fontsize=24)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(rns, stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(rns, stations, rotation=0)

    return fig
