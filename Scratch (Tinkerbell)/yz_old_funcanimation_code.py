'''done:
plot stations
plot vector readings + gifs

'''


'''todo:
plot connections
auto ortho_trans based on stations plotted
colour code arrows to improve readability - map long, lat onto 2d grid of colours


improve efficiency of gif fn if possible
make gifs
remove redundancies in plot_stations and plot_data_globe
incorporate MLAT, MLT as outlined by IGRF? make sure same version as kendal and all other data
'''




%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xarray as xr

import lib.supermag as sm
import lib.supermagyz as yz

pwd()

station_readings = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)


# yz.plot_stations(ds1.station, ortho_trans)


list_of_stations = station_readings.station




t = station_readings.time[1]

ortho_trans = np.array([-100, 50])

test = yz.plot_data_globe(station_readings, t, list_of_stations, ortho_trans)
test.savefig(fname = "test")
test.quivers = []


test.ax
test

import sys
from matplotlib.animation import FuncAnimation

# Plot a scatter that persists (isn't redrawn) and the initial line.
station_coords = yz.csv_to_coords("First Pass/20190420-12-15-supermag-stations.csv")








fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1]))
# fig = plt.figure(figsize = (20, 20))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
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

line, = ax.quiver(np.array([]), np.array([]), np.array([]), np.array([]), transform = ccrs.PlateCarree(), width = 0.002, color = "g")



def init():
    # plswork = yz.plot_stations(list_of_stations, ortho_trans)



    station_coords = csv_to_coords("First Pass/20190420-12-15-supermag-stations.csv")
    # fig = plt.figure(figsize = (20, 20))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
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





    return ax




def animate(i):
    station_readings = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                                MLT = True, MLAT = True)
    station_coords = yz.csv_to_coords("First Pass/20190420-12-15-supermag-stations.csv")

    list_of_stations = station_readings.station
    t = station_readings.time[i]
    # return yz.plot_data_globe(station_readings, t, list_of_stations, np.array([-100, 50])),

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

    # fig = plt.figure(figsize = (20, 20))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(ortho_trans[0], ortho_trans[1])) #(long, lat)
    # ax.add_feature(cfeature.OCEAN, zorder=0)
    # ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    # ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    # ax.add_feature(cfeature.LAKES, zorder=0)
    # ax.set_global()
    # ax.gridlines()

    # plswork.scatter(x, y, transform = ccrs.Geodetic()) #plots stations
    ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
              width = 0.002, color = "g")

    return ax






anim = FuncAnimation(fig, animate, frames=10, interval=20, blit = True)

plt.show()

anim.save('First Pass/test.mp4', fps = 3)

























import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import lib.supermagyz as yz
import lib.supermag as sm
from matplotlib.animation import FuncAnimation

from PIL import Image


ortho_trans = (-100, 40)

fig = plt.figure(figsize = (15, 15))
# ax = fig.axes(projection=ccrs.Orthographic(0, 0))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-100, 40))
ax.add_feature(cfeature.OCEAN, zorder=0)
ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
ax.add_feature(cfeature.LAKES, zorder=0)
ax.set_global()
ax.gridlines()

station_coords = yz.csv_to_coords("First Pass/20190420-12-15-supermag-stations.csv")
station_readings = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

list_of_stations = station_readings.station
num_stations = len(list_of_stations)
x = np.zeros(num_stations)
y = np.zeros(num_stations)
u = np.zeros(num_stations)
v = np.zeros(num_stations)
i = 0

t = station_readings.time[0]
for station in list_of_stations:
    x[i] = station_coords.longitude.loc[dict(station = station)]
    y[i] = station_coords.latitude.loc[dict(station = station)]
    u[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "E")]
    v[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "N")]
    i += 1

Q = ax.scatter(x, y, transform = ccrs.Geodetic()) #plots stations
# Q = ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
          # width = 0.002, color = "g")

fig.savefig("tst.png")
# plt.savefig("test.png")
# Q.lines


blah = len(station_readings.time)

for i in range(blah):
    t = station_readings.time[i]
    fig = yz.plot_data_globe(station_readings, t, list_of_stations, ortho_trans)
    fig.savefig("gif/%s.png" %i)

names = []

for i in range(500):
    names.append("gif/%s.png" %i)

images = []


for n in names:
    frame = Image.open(n)
    images.append(frame)

images[0].save("gif/500frames.gif", save_all = True, append_images = images[1:], duration = 50, loop = 0)




def update_quiver(num, Q, x, y):
    station_coords = yz.csv_to_coords("First Pass/20190420-12-15-supermag-stations.csv")
    num_stations = len(list_of_stations)
    x = np.zeros(num_stations)
    y = np.zeros(num_stations)
    u = np.zeros(num_stations)
    v = np.zeros(num_stations)
    i = 0
    t = station_readings.time[num]

    for station in list_of_stations:
        x[i] = station_coords.longitude.loc[dict(station = station)]
        y[i] = station_coords.latitude.loc[dict(station = station)]
        u[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "E")]
        v[i] = station_readings.measurements.loc[dict(station = station, time = t, reading = "N")]
        i += 1

    # Q.cla()
    # plt.cla()
    # plt.cla()
    # fig = plt.figure(figsize = (15, 15))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-100, 40))
    # ax.add_feature(cfeature.OCEAN, zorder=0)
    # ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
    # ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
    # ax.add_feature(cfeature.LAKES, zorder=0)
    # ax.set_global()
    # ax.gridlines()




    Q = ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), #plots vector data
              width = 0.002, color = "g")

    return Q,

anim = FuncAnimation(fig, update_quiver, fargs = (Q, x, y), interval = 50, blit = False)

anim.save('First Pass/test.mp4')#, extra_args = {"frameon": True})




# # line, = ax.plot([], [], lw=2)
#
# def init():
#     line.set_data([], [])
#     return line,
#
# def animate(i):
#
#     # ax.add_feature(cfeature.OCEAN, zorder=0)
#     # ax.add_feature(cfeature.LAND, zorder=0, edgecolor='grey')
#     # ax.add_feature(cfeature.BORDERS, zorder=0, edgecolor='grey')
#     # ax.add_feature(cfeature.LAKES, zorder=0)
#     # ax.set_global()
#     # ax.gridlines()
#     line.set_data(x, y)
#     return line,
#
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=100, interval=20, blit=True)
#
# anim.save("test.mp4", fps=30, extra_args = ["-vcodec", "libx264"])
