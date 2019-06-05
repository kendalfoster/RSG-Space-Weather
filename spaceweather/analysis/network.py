"""
Contents
--------

- degree
- num_edges
- cluster_coeff
"""

## Packages
import numpy as np
import xarray as xr # if gives error, just rerun



def degree(adj_matrix, avg=False, norm=True):
    '''
    Find the degree of each station in the network for each time.
    Optionally, return the average (over time) degree of each station.

    Parameters
    ----------
    adj_matrix : xarray.Dataset
        Output of :func:`spaceweather.analysis.threshold.adjMat`.
    avg : bool, optional
        Return the average (over time) degree of each station. Default is False.
    norm : bool, optional
        Whether or not you want the average degree to be normalized in [0,1].
        Default is True.

    Returns
    -------
    xarray.Dataset
        Data_vars are: degree, avg_deg.\n
        Coordinates are: win_start, station.
    '''

    # constants
    stations = adj_matrix.first_st.values
    nsta = len(stations)
    wins = adj_matrix.win_start
    ntime = len(wins)

    # initialize array
    degs = np.full(shape = (ntime, nsta), fill_value = np.nan)

    # loop through each time
    for t in range(ntime):
        am = adj_matrix[dict(win_start = t)]
        # sum up connections for each station
        for i in range(nsta):
            degs[t,i] = np.sum(am[dict(first_st = i)].adj_coeffs).values + np.sum(am[dict(second_st = i)].adj_coeffs).values

    if norm:
        degs = degs/(nsta-1)

    # construct Dataset
    ds = xr.Dataset(data_vars = {'degree': (['win_start', 'station'], degs)},
                    coords = {'win_start': wins,
                              'station': stations})

    # optionally add average degree (over time)
    if avg:
        avg_degs = np.nanmean(degs, axis=0)
        ds_ad = xr.Dataset(data_vars = {'avg_deg': (['station'], avg_degs)},
                           coords = {'station': stations})
        ds = xr.merge([ds, ds_ad])

    return ds


def num_edges(adj_matrix, avg=False, norm=True):
    '''
    Find the number of edges in the network for each time.
    Optionally, return the average (over time) number of edges in the network.

    Parameters
    ----------
    adj_matrix : xarray.Dataset
        Output of :func:`spaceweather.analysis.threshold.adjMat`.
    avg : bool, optional
        Return the average (over time) degree of each station. Default is False.
    norm : bool, optional
        Whether or not you want the average degree to be normalized in [0,1].
        Default is True.

    Returns
    -------
    xarray.Dataset
        Data_vars are: edges, avg_edge.\n
        Coordinates are: win_start.
    '''

    # constants
    stations = adj_matrix.first_st.values
    nsta = len(stations)
    wins = adj_matrix.win_start
    ntime = len(wins)

    # initialize array
    nedges = np.full(shape = ntime, fill_value = np.nan)

    # loop through each time
    for t in range(ntime):
        nedges[t] = np.sum(adj_matrix[dict(win_start = t)].adj_coeffs).values

    if norm:
        nedges = nedges/(nsta*(nsta-1))

    # construct Dataset
    ds = xr.Dataset(data_vars = {'edges': (['win_start'], nedges)},
                    coords = {'win_start': wins})

    # optionally add average degree (over time)
    if avg:
        avg_edges = np.nanmean(nedges, axis=0)
        ds_ad = xr.Dataset(data_vars = {'avg_edge': ([], avg_edges)})
        ds = xr.merge([ds, ds_ad])

    return ds


def cluster_coeff(adj_matrix):
    '''
    Find the average local clustering coefficient of each station in the network,
    over time. Also calculates the network average clustering coefficient, also
    averaged over time.

    Parameters
    ----------
    adj_matrix : xarray.Dataset
        Output of :func:`spaceweather.analysis.threshold.adjMat`.

    Returns
    -------
    xarray.Dataset
        Data_vars are: local_cc, net_avg_cc.\n
        Coordinates are: station.
    '''

    # constants
    stations = adj_matrix.first_st.values
    nsta = len(stations)
    ntime = len(adj_matrix.win_start)

    # initialize arrays
    cc = np.full(shape = (ntime, nsta), fill_value = np.nan)

    # loop through each time
    for t in range(ntime):
        am = adj_matrix[dict(win_start = t)]
        # loop through each station
        for i in range(nsta):
            # find station i's connections
            nbrs = []
            for j in range(nsta):
                if am[dict(first_st = i, second_st = j)].adj_coeffs.values == 1 or am[dict(first_st = j, second_st = i)].adj_coeffs.values == 1:
                    nbrs.append(j)

            # find the degree of station i
            deg = len(nbrs)

            if deg > 1:
                # find the number of edges between its neighbors
                nbr_edges = 0
                for n1 in range(len(nbrs)-1):
                    for n2 in range(1, len(nbrs)):
                        if am[dict(first_st = n1, second_st = n2)].adj_coeffs.values == 1 or am[dict(first_st = n2, second_st = n1)].adj_coeffs.values == 1:
                            nbr_edges += 1

                # calculate local clustering coefficient
                cc[t,i] = 2*nbr_edges/(deg*(deg-1))

    # calculate average local clustering coefficient for each station
    local_cc = np.zeros(nsta)
    for i in range(nsta):
        local_cc[i] = np.nanmean(cc[:,i])

    # calculate network average clustering coefficient
    net_avg_cc = np.nanmean(local_cc)

    # construct DataArrays
    local_cc_da = xr.DataArray(data = local_cc,
                               coords = [stations],
                               dims = ['station'])
    net_avg_cc_da = xr.DataArray(data = net_avg_cc)

    # merge DataArrays into one Dataset
    ds = xr.merge([local_cc_da.rename('local_cc'),
                  net_avg_cc_da.rename('net_avg_cc')])

    return ds
