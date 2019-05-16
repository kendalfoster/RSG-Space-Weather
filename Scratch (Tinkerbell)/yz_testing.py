'''done:
plot stations
plot vector readings + gifs
plot connections (partial)
auto ortho_trans based on stations plotted
'''


'''todo:
plot connections - list_of_stations, t
colour code arrows to improve readability - map long, lat onto 2d grid of colours
colour code vertex connections similar to Dods

improve efficiency of gif fn if possible
make gifs
remove redundancies in plot_stations and plot_data_globe
incorporate MLAT, MLT as outlined by IGRF? make sure same version as kendal and all other data
'''



'''before running install cartopy using "conda install -c conda-forge cartopy" '''


from cartopy.feature.nightshade import Nightshade
# import spaceweather.analysis.supermag as sm
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
# import spaceweather.vi
from spaceweather.visualisation import globes as svg
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

station_readings = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

test = sac.cca(station_readings)

t = station_readings.time[1]

t.data


plot_data_globe_colour(station_readings, t.data)



def plot_data_globe_colour(station_readings, t, list_of_stations = None, ortho_trans = (0, 0)):
    if np.all(list_of_stations == None):
        list_of_stations = station_readings.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = svg.auto_ortho(list_of_stations)

    station_coords = svg.csv_to_coords()
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
    ax.add_feature(Nightshade(t))
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
