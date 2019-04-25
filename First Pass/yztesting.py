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



import lib.supermagyz as yz
import lib.supermag as sm
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import cartopy.feature as cfeature

station_readings = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

t = station_readings.time[1]
list_of_stations = station_readings.station

new_list = yz.csv_to_coords().station


yz.plot_data_globe(station_readings, t, list_of_stations = None, ortho_trans = (0, 0))
# plots N and E components of the vector readings for a single time step t
# by default it plots data from all stations fed to it in station_readings unless
# specified otherwise in list_of_stations.
# ortho_trans specifies the angle from which we see the plot(earth) at.
# if left at default, yz.auto_ortho(list_of_stations) centres the view on the centre of all stations in list_of_stations.




yz.data_globe_gif(station_readings, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra")
#makes sandra.gif in the /gif folder


#generating fake adjacency matrix
N = 20
# length = 50
b = np.random.randint(-2000,2000,size=(N,N))


b_symm = (b + b.T)/2

fake_data = b_symm < 0

yz.plot_connections_globe(station_readings, adj_matrix = fake_data[:10, :10], ortho_trans = (0, 0), t = None, list_of_stations = new_list[:10])
#plots connections between stations.
#for now it expects a 2d adjacency matrix as input but i will add code to make it do 3d(time on 3rd axis) as well




def plot_data_globe(station_readings, t, list_of_stations = None, ortho_trans = (0, 0)):
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


np.array((1, 1)).shape

test = np.zeros((1, 5))

test = (1, 2, 3)

test[2:3]

plot_data_globe(station_readings, t)


station_coords = yz.csv_to_coords()

max(station_coords.longitude)

plc.hsv_to_rgb((1, 1, 1))
test = np.ones((2, 3))
plc.hsv_to_rgb(test)


num_stations = len(list_of_stations)
colours = np.ones((num_stations, 3))
colours
for i in range(num_stations):
    colours[i, 0] = station_coords.longitude.loc[dict(station = list_of_stations[i])]/360
    colours[i, 2] = (station_coords.latitude.loc[dict(station = list_of_stations[i])]+90)/180

colours = plc.hsv_to_rgb(colours)
