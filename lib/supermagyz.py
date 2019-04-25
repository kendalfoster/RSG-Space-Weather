import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from PIL import Image

def csv_to_coords():
    csv_file = "First Pass/20190420-12-15-supermag-stations.csv"
    stationdata = pd.read_csv(csv_file, usecols = [0, 1, 2])

    IAGAs = stationdata["IAGA"]
    LATs = stationdata["GEOLAT"]
    LONGs = stationdata["GEOLON"]
    data = xr.Dataset(data_vars = {"latitude": (["station"], LATs), "longitude": (["station"], LONGs)}, coords = {"station": list(IAGAs)})
    for long in data.longitude:
        if long < 0:
            long += 360

    return data

def auto_ortho(list_of_stations):
    station_coords = csv_to_coords()
    av_long = sum(station_coords.longitude.loc[dict(station = s)] for s in list_of_stations)/len(list_of_stations)
    av_lat = sum(station_coords.latitude.loc[dict(station = s)] for s in list_of_stations)/len(list_of_stations)

    return np.array((av_long, av_lat))

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

def plot_data_globe(station_readings, t, list_of_stations = None, ortho_trans = (0, 0)):
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
        u[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "E")]
        v[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "N")]
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
        ortho_trans = yz.auto_ortho(list_of_stations)

    station_coords = yz.csv_to_coords()
    num_stations = len(list_of_stations)
    x = np.zeros(num_stations)
    y = np.zeros(num_stations)
    u = np.zeros(num_stations)
    v = np.zeros(num_stations)
    i = 0

    for station in list_of_stations:
        x[i] = station_coords.longitude.loc[dict(station = station)]
        y[i] = station_coords.latitude.loc[dict(station = station)]
        u[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "E")]
        v[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "N")]
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
        colours[i, 2] = (station_coords.latitude.loc[dict(station = list_of_stations[i])]+90)/180

    colours = plc.hsv_to_rgb(colours)

    for i in range(num_stations):
        ax.quiver(x[i:i+1], y[i:i+1], u[i:i+1], v[i:i+1], transform = ccrs.PlateCarree(), #plots vector data
              width = 0.002, color = colours[i, :])

    return fig

def data_globe_gif(station_readings, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra"):
    #times in terms of index in the array, might be helpful to have a fn to look up index from timestamps
    names = []
    images = []
    list_of_stations = station_readings.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = auto_ortho(list_of_stations)

    for i in range(time_start, time_end):
        t = station_readings.time[i]
        fig = plot_data_globe(station_readings, t, list_of_stations, ortho_trans)
        fig.savefig("gif/images_for_giffing/%s.png" %i)

    for i in range(time_start, time_end):
        names.append("gif/images_for_giffing/%s.png" %i)

    for n in names:
        frame = Image.open(n)
        images.append(frame)

    images[0].save("gif/%s.gif" %file_name, save_all = True, append_images = images[1:], duration = 50, loop = 0)

def plot_connections_globe(station_readings, adj_matrix, ortho_trans = (0, 0), t = None, list_of_stations = None):
    '''right now this assumes i want to plot all stations in the adj_matrix for a single time,
       will add more later
       also gives 2 plots for some reason'''

    if np.all(list_of_stations == None):
        list_of_stations = station_readings.station

    if np.all(ortho_trans == (0, 0)):
        ortho_trans = auto_ortho(list_of_stations)

    if t == None:
        num_sta = len(adj_matrix)
        fig = plot_stations(list_of_stations, ortho_trans)
        station_coords = csv_to_coords()
        ax = fig.axes[0]

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

    return fig
