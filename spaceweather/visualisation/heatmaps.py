## Packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca


def plot_mag_adj_mat(ds, ds_win, n0=0.25, components=['N', 'E', 'Z']):
    """
    Calculate and plot the adjacency matrix for a set of stations during one time window.

    This function does the same as :func:`supermag.mag_adj_mat`. In addition to
    calculating the adjacency matrix, this also returns the plot of the
    adjacency matrix.

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
    components : list, optional
        List of components in the data. Default is ['N', 'E', 'Z'].

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

    cca = inter_st_cca(ds=ds_win, components= components)
    cca = cca.assign_coords(first_st = range(num_st))
    cca = cca.assign_coords(second_st = range(num_st))

    thresh = mag_thresh_dods(ds=ds, n0=n0, components=components)
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

    return adj_mat, fig


def corellogram(ds, station1, station2, lag_range=10, win_len=128):
    """
    Calculate and plots a corllogram for two stations
    
    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This is used to calculate the correlations
    station1 and station2:
        Stations you want to have a corellogram comparing, station1 remains fixed whilst
        the window is shifted for station2
    lag_range: float, default 10
        The range of lags you want to examine
    win_len: float, default 128
        The length you want your window to be

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the corellogram
    """

    #Window the data
    windowed = window(ds,win_len)

    #Generating appropriate dimensions for our array
    a = windowed.measurements.loc[dict(station = station1)].loc[dict(component = "N")][:,0]
    time_length = len(a)
    time_range = time_length - 2 * lag_range

    x = np.arange(time_range) + lag_range + 1
    y = np.arange(2*lag_range+1) - lag_range
    z = np.zeros([len(y),time_range])

    #Do correlations
    for i in range(len(y)):
        for j in range(time_range):
            z[i,j] = inter_phase_dir_corr(ds,station1,station2,x[j]-1,y[i]+x[j]-1,win_len,components=None)


    #Produce heatmap
    plot = plt.pcolormesh(x,y,z)

    return x, y , z
