## Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import pandas as pd
import xarray as xr # if gives error, just rerun
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Local Packages
# import spaceweather.rcca as rcca


## Dependencies
# numpy
# scipy
# matplotlib.pyplot
# pandas
# xarray
# cartopy
# rcca (code downloaded from GitHub)

## Notes
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html

## Unused Packages, but potentially useful
# import xscale.signal.fitting as xsf # useful functions for xarray data structures
    # pip3 install git+https://github.com/serazing/xscale.git
    # pip3 install toolz




################################################################################
####################### Visualizing The Network ################################

##
def csv_to_coords():
    csv_file = "Data/station_coords.csv"
    stationdata = pd.read_csv(csv_file, usecols = [0, 1, 2])

    IAGAs = stationdata["IAGA"]
    LATs = stationdata["GEOLAT"]
    LONGs = stationdata["GEOLON"]
    data = xr.Dataset(data_vars = {"latitude": (["station"], LATs), "longitude": (["station"], LONGs)}, coords = {"station": list(IAGAs)})

    return data


##
def auto_ortho(list_of_stations):
    station_coords = csv_to_coords()
    av_long = sum(station_coords.longitude.loc[dict(station = s)] for s in list_of_stations)/len(list_of_stations)
    av_lat = sum(station_coords.latitude.loc[dict(station = s)] for s in list_of_stations)/len(list_of_stations)

    return np.array((av_long, av_lat))


##
def plot_stations(list_of_stations, ortho_trans):
    station_coords = csv_to_coords()
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.LAKES, zorder=0)
    ax.set_global()
    ax.gridlines()

    num_sta = len(list_of_stations)
    longs = np.zeros(num_sta)
    lats = np.zeros(num_sta)
    for i in range(num_sta):
        longs[i] = station_coords.longitude.loc[dict(station = list_of_stations[i])]
        lats[i] = station_coords.latitude.loc[dict(station = list_of_stations[i])]
    ax.scatter(longs, lats, transform = ccrs.Geodetic())

    return fig


##
def plot_data_globe(station_components, t, list_of_stations = None, ortho_trans = (0, 0)):
    if np.all(list_of_stations == None):
        list_of_stations = station_components.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = auto_ortho(list_of_stations)

    station_coords = csv_to_coords()
    num_stations = len(list_of_stations)
    x = np.zeros(num_stations)
    y = np.zeros(num_stations)
    u = np.zeros(num_stations)
    v = np.zeros(num_stations)
    i = 0

    for station in list_of_stations:
        x[i] = station_coords.longitude.loc[dict(station = station)]
        y[i] = station_coords.latitude.loc[dict(station = station)]
        u[i] = station_components.measurements.loc[dict(station = station, time = t, component = "E")]
        v[i] = station_components.measurements.loc[dict(station = station, time = t, component = "N")]
        i += 1

    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.LAKES, zorder=0)
    ax.set_global()
    ax.gridlines()

    ax.scatter(x, y, transform = ccrs.Geodetic()) #plots stations
    ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
              width = 0.002, color = "g")

    return fig


def plot_data_globe_colour(station_readings, t, list_of_stations = None, ortho_trans = (0, 0)):
    if np.all(list_of_stations == None):
        list_of_stations = station_readings.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = auto_ortho(list_of_stations)

    station_coords = csv_to_coords()
    num_stations = len(list_of_stations)
    x = np.zeros(num_stations)
    y = np.zeros(num_stations)
    u = np.zeros(num_stations)
    v = np.zeros(num_stations)
    i = 0

    for station in list_of_stations:
        x[i] = station_coords.longitude.loc[dict(station = station)]
        y[i] = station_coords.latitude.loc[dict(station = station)]
        u[i] = station_readings.measurements.loc[dict(station = station, time = t, component = "E")]
        v[i] = station_readings.measurements.loc[dict(station = station, time = t, component = "N")]
        i += 1

    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    ax.add_feature(cfeature.LAKES, zorder=0)
    ax.set_global()
    ax.gridlines()

    ax.scatter(x, y, color = "k", transform = ccrs.Geodetic()) #plots stations

    colours = np.ones((num_stations, 3))

    for i in range(num_stations):
        colours[i, 0] = station_coords.longitude.loc[dict(station = list_of_stations[i])]/360
        colours[i, 2] = (station_coords.latitude.loc[dict(station = list_of_stations[i])]-10)/80

    colours = plc.hsv_to_rgb(colours)

    ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
          width = 0.002, color = colours)

    ts = pd.to_datetime(str(t.data))
    mytime = ts.strftime('%Y.%m.%d %H:%M')

    plt.title("%s" %mytime, fontsize = 30)

    return fig



##
def plot_connections_globe(station_components, adj_matrix, ortho_trans = (0, 0), t = None, list_of_stations = None):
    '''right now this assumes i want to plot all stations in the adj_matrix for a single time,
       will add more later
       also gives 2 plots for some reason'''

    if list_of_stations == None:
        list_of_stations = station_components.station

    if np.all(ortho_trans == (0, 0)):
        ortho_trans = auto_ortho(list_of_stations)

    if t == None:
        num_sta = len(adj_matrix)
        fig = plot_stations(station_components.station, ortho_trans)
        station_coords = csv_to_coords()
        ax = fig.axes[0]

        for i in range(num_sta-1):
            for j in range(i+1, num_sta):
                if adj_matrix[i, j] == 1:
                    station_i = station_components.station[i]
                    station_j = station_components.station[j]
                    long1 = station_coords.longitude.loc[dict(station = station_i)]
                    long2 = station_coords.longitude.loc[dict(station = station_j)]
                    lat1 = station_coords.latitude.loc[dict(station = station_i)]
                    lat2 = station_coords.latitude.loc[dict(station = station_j)]

                    ax.plot([long1, long2], [lat1, lat2], color='blue', transform=ccrs.Geodetic())

    return fig



################################################################################
