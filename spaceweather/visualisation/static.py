"""
Contents
--------

- csv_to_coords
- auto_ortho
- plot_stations
- plot_data_globe
- plot_connections_globe
- plot_lag_network
"""


## Packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import pandas as pd
import xarray as xr # if gives error, just rerun
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
# proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share') # Windows
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj') # Linux
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap



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


def plot_stations(list_of_stations, ortho_trans, sta_col='black', **kwargs):
    '''
    Plot the stations on a globe.

    Parameters
    ----------
    list_of_stations : list
        List of stations in ds to be used on the plot.
    ortho_trans : tuple
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.
    sta_col : str, optional
        Color for the plotted stations. Default is black.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of the network on the globe.
    '''

    # check kwargs
    s_c = kwargs.get('station_coords', None)
    if s_c is not None:
        station_coords = s_c
    else:
        station_coords = csv_to_coords()
    c_l = kwargs.get('sta_col', None)
    if c_l is not None:
        sta_col = c_l

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
    ax.scatter(longs, lats, transform = ccrs.Geodetic(), c = sta_col)

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
            raise ValueError('Error: need to input either\n 1) ds\n 2) list_of_stations and time')
        if time is None:
            raise ValueError('Error: need to input either\n 1) ds\n 2) list_of_stations and time')
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


def plot_lag_network(adj_matrix, lag_range=10, ortho_trans=None,
                     sta_color='black', daynight=True):
    '''
    Plot the directed network provided by the adjacency matrix and lag.

    Parameters
    ----------
    adj_matrix : xarray.Dataset
        Adjacency matrix for the network; output from :func:`spaceweather.analysis.threshold.adj_mat`.
    lag_range: int, optional
        The range, in minutes, of positive and negative shifts for the second station in each pair.
    ortho_trans : tuple, optional
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.
    sta_color : str, optional
        Color for the plotted stations. Default is black.
    daynight : bool, optional
        Whether or not to include a shadow for nighttime. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of directed network with lag values as colors.
    '''

    # constants
    time = pd.to_datetime(adj_matrix.win_start.values)
    list_of_sta = adj_matrix.first_st
    num_sta = len(list_of_sta)
    sta_coords = csv_to_coords()
    if ortho_trans is None:
        ortho_trans = auto_ortho(list_of_sta)

    # initialize plot
    fig = plt.figure(figsize=(20, 20))
    map = Basemap(projection='ortho', lat_0=ortho_trans[1], lon_0=ortho_trans[0])
    map.drawmapboundary(fill_color='lightsteelblue')
    map.fillcontinents(color='beige', lake_color='lightsteelblue', zorder=0)
    map.drawcoastlines(color='darkslategrey', zorder=1)
    map.drawcountries(color='darkslategrey', zorder=1)
    map.drawmeridians(np.arange(0,360,30), color='darkgrey', zorder=0)
    map.drawparallels(np.arange(-90,90,30), color='darkgrey', zorder=0)

    # get lon/lat data for each station
    lons = np.zeros(num_sta)
    lats = np.zeros(num_sta)
    for i in range(num_sta):
        lons[i] = sta_coords.longitude.loc[dict(station = list_of_sta[i])]
        lats[i] = sta_coords.latitude.loc[dict(station = list_of_sta[i])]

    # draw stations on map
    lons, lats = map(lons, lats)
    map.scatter(lons, lats, color=sta_color)

    # if connected, get the lag between each station pair
    x = []
    y = []
    u = []
    v = []
    lags = []
    for i in range(num_sta-1):
        for j in range(i+1, num_sta):
            a_m = adj_matrix[dict(first_st = i, second_st = j)]
            if a_m.adj_coeffs.values == 1:
                lg = a_m.lag.values
                if lg < 0:
                    # append to scatter lists
                    x.append(lons[j])
                    y.append(lats[j])
                    u.append(lons[i]-lons[j])
                    v.append(lats[i]-lats[j])
                    lags.append(-lg)
                else:
                    # append to scatter lists
                    x.append(lons[i])
                    y.append(lats[i])
                    u.append(lons[j]-lons[i])
                    v.append(lats[j]-lats[i])
                    lags.append(lg)

    # plot the lag arrows on map
    norm = plc.Normalize(0, lag_range)
    cmap = matplotlib.cm.viridis
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    map.quiver(x, y, u, v, lags, cmap=cmap, norm=norm, angles='xy', scale_units='xy', scale=1, width=0.002)
    cbar = map.colorbar(sm)
    cbar.set_label('Lag, minutes')

    # add nighttime shadow
    if daynight:
        map.nightshade(time, alpha=0.2)

    # add timestamp as plot title
    mytime = time.strftime('%Y.%m.%d %H:%M')
    plt.title('Lag Network at ''%s' %mytime, fontsize = 30)

    return fig
