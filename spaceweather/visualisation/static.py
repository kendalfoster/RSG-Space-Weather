"""
Contents
--------

- csv_to_coords
- auto_ortho
- plot_stations
- plot_data_globe
- plot_connections_globe
"""


## Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import pandas as pd
import xarray as xr # if gives error, just rerun
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade


## Notes
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html



def csv_to_coords():
    '''
    Get the latitude and longitude for each station in the SuperMAG database.

    Parameters
    ----------

    Returns
    -------
    xarray.Dataset
        The data_vars are: latitude, longitude.\n
        The coordinates are: station.
    '''

    # read csv file containing the station coordinates
    csv_file = "Data/station_coords.csv"
    stationdata = pd.read_csv(csv_file, usecols = [0, 1, 2])

    # extract the latitude and longitude for each station
    IAGAs = stationdata["IAGA"]
    LATs = stationdata["GEOLAT"]
    LONGs = stationdata["GEOLON"]

    # create a Dataset
    data = xr.Dataset(data_vars = {"latitude": (["station"], LATs),
                                   "longitude": (["station"], LONGs)},
                      coords = {"station": list(IAGAs)})

    return data


def auto_ortho(list_of_stations):
    '''
    Get the average latitude and average longitude for each station.

    Parameters
    ----------
    list_of_stations : list
        List of stations in ds to be used on the plot.

    Returns
    -------
    tuple
        Orientation of the plotted globe; determines the angle at which we view the globe.
    '''

    station_coords = csv_to_coords()
    av_long = sum(station_coords.longitude.loc[dict(station = s)] for s in list_of_stations)/len(list_of_stations)
    av_lat = sum(station_coords.latitude.loc[dict(station = s)] for s in list_of_stations)/len(list_of_stations)

    return np.array((av_long, av_lat))


def plot_stations(list_of_stations, ortho_trans, **kwargs):
    '''
    Plot the stations on a globe.

    Parameters
    ----------
    list_of_stations : list
        List of stations in ds to be used on the plot.
    ortho_trans : tuple
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the network on the globe.
    '''

    # check if kwargs contains station_coords
    s_c = kwargs.get('station_coords', None)
    if s_c is not None:
        station_coords = s_c
    else:
        station_coords = csv_to_coords()

    # initialize plot of globe with features
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.LAKES, zorder=0)
    ax.set_global()
    ax.gridlines()

    # get latitudes and longitudes of stations
    num_sta = len(list_of_stations)
    longs = np.zeros(num_sta)
    lats = np.zeros(num_sta)
    for i in range(num_sta):
        longs[i] = station_coords.longitude.loc[dict(station = list_of_stations[i])]
        lats[i] = station_coords.latitude.loc[dict(station = list_of_stations[i])]

    # add stations to plot
    ax.scatter(longs, lats, transform = ccrs.Geodetic())

    return fig


def plot_data_globe(ds, list_of_stations=None, list_of_components=['N', 'E'],
                     t=0, ortho_trans=None, daynight=True, colour=False, **kwargs):
    '''
    Plot the data as vectors for each station on a globe for a single time
    with an optional shadow for nighttime and optional data colouration.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    list_of_stations : list, optional
        List of stations in ds to be used on the plot.
    list_of_components : list, optional
        List of components in ds to be used on the plot. Must be of length 2.
    t : int or numpy.datetime64, optional
        Either\n
        1) the index of the time in ds, or\n
        2) the long-format time time in ds\n
        to be used to plot the data. Defaults to the first time in ds.
    ortho_trans : tuple, optional
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.
    daynight : bool, optional
        Whether or not to include a shadow for nighttime. Default is True.
    colour : bool, optional
        Whether or not to colour the data vectors. Also accepts 'color' for
        Americans who can't spell properly.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the network on the globe.
    '''

    # check kwargs for color
    cl = kwargs.get('color', None)
    if cl is not None:
        colour = cl

    # check inputs
    if len(list_of_components) != 2:
        print('Error: please input two components in list_of_components')
        return 'Error: please input two components in list_of_components'
    if list_of_stations is None:
        list_of_stations = ds.station.values
    if ortho_trans is None:
        ortho_trans = auto_ortho(list_of_stations)
    if isinstance(t, int):
        t = ds[dict(time = t)].time.values # extract time at t

    # get constants
    station_coords = csv_to_coords()
    num_stations = len(list_of_stations)

    # store latitude and longitude of the stations
    x = station_coords.longitude.loc[dict(station = list_of_stations)].values
    y = station_coords.latitude.loc[dict(station = list_of_stations)].values

    # store measurements for each coordinate
    u = ds.measurements.loc[dict(time = t,
                                 station = list_of_stations,
                                 component = list_of_components[1])].values
    v = ds.measurements.loc[dict(time = t,
                                 station = list_of_stations,
                                 component = list_of_components[0])].values

    # create figure
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.LAKES, zorder=0)
    ax.set_global()
    ax.gridlines()

    # plot stations and measurement vectors
    ax.scatter(x, y, color = "k", transform = ccrs.Geodetic()) #plots stations
    if colour:
        colours = np.ones((num_stations, 3))
        colours[:, 0] = x/360
        colours[:, 2] = (y-10)/80
        colours = plc.hsv_to_rgb(colours)
        ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
              width = 0.002, color = colours)
    else:
        ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
                  width = 0.002, color = "g")

    # add shadow for nighttime
    if daynight:
        ax.add_feature(Nightshade(pd.to_datetime(t)), alpha = 0.2)

    # add timestamp as plot title
    dt = pd.to_datetime(t)
    mytime = dt.strftime('%Y.%m.%d %H:%M')
    plt.title("%s" %mytime, fontsize = 30)

    return fig


def plot_connections_globe(adj_matrix, ds=None, list_of_stations=None, time=None,
                           ortho_trans=None, daynight=True, **kwargs):
    '''
    Plot the network on a globe for a single time with an optional shadow for nighttime.

    Parameters
    ----------
    ds : xarray.Dataset, optional
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
        If ds is not included then list_of_stations and time must be included.
    list_of_stations : list, optional
        List of stations in ds to be used on the plot.
        If list_of_stations is not included then ds must be included.
    time : numpy.datetime64, optional
        The time and date for the adj_matrix.
        If time is not included then ds must be included, and time defaults to
        the first time in ds.
    adj_matrix : numpy.ndarray
        The adjacency matrix for the connections between stations.
    ortho_trans : tuple, optional
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.
    daynight : bool, optional
        Whether or not to include a shadow for nighttime. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the network on the globe.
    '''

    # check inputs
    if ds is None:
        if list_of_stations is None:
            print('Error: need to input either\n 1) ds\n 2) list_of_stations and time')
            return 'Error: need to input either\n 1) ds\n 2) list_of_stations and time'
        if time is None:
            print('Error: need to input either\n 1) ds\n 2) list_of_stations and time')
            return 'Error: need to input either\n 1) ds\n 2) list_of_stations and time'
    else:
        list_of_stations = ds.station
        time = pd.to_datetime(ds.time.values[0])
    time = pd.to_datetime(time)

    if ortho_trans is None:
        ortho_trans = auto_ortho(list_of_stations)

    # get constants
    num_sta = len(list_of_stations)
    station_coords = csv_to_coords()

    # initialize plot
    fig = plot_stations(list_of_stations, ortho_trans,
                        station_coords = station_coords, **kwargs)
    ax = fig.axes[0]

    # if connected, plot the connection between each station pair
    for i in range(num_sta-1):
        for j in range(i+1, num_sta):
            if adj_matrix[i, j] == 1:
                station_i = list_of_stations[i]
                station_j = list_of_stations[j]
                long1 = station_coords.longitude.loc[dict(station = station_i)]
                long2 = station_coords.longitude.loc[dict(station = station_j)]
                lat1 = station_coords.latitude.loc[dict(station = station_i)]
                lat2 = station_coords.latitude.loc[dict(station = station_j)]

                ax.plot([long1, long2], [lat1, lat2], color='blue', transform=ccrs.Geodetic())

    # add shadow for nighttime
    if daynight:
        ax.add_feature(Nightshade(time), alpha = 0.2)

    # add timestamp as plot title
    mytime = time.strftime('%Y.%m.%d %H:%M')
    plt.title("%s" %mytime, fontsize = 30)

    return fig
