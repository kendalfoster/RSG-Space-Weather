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


# import spaceweather.analysis.supermag as sm
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.flocking as saf
# import spaceweather.vi
from spaceweather.visualisation import globes as svg
import spaceweather.visualisation.animations as sva
import spaceweather.visualisation.lines as svl
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import xarray as xr

import pandas as pd

data = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
original = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
data_ms = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

data2 = sad.mag_csv_to_Dataset(csv_file = "Data/20010305-16-38-supermag.csv",
                            MLT = True, MLAT = True)

N110 = sad.mag_csv_to_Dataset(csv_file = "Data/N110.csv",
                            MLT = True, MLAT = True)

N110_ms = sad.mag_csv_to_Dataset(csv_file = "Data/N110.csv",
                            MLT = True, MLAT = True)


test = sad.meansubtract(N110)

plt.figure(figsize = (20, 8))
plt.plot(test.measurements.loc[dict(station = "NAL", time = test.time[range(1500)])])









N110.measurements.loc[dict(station = "NAL", component = "N")]

running_mean = N110.measurements.loc[dict(station = "NAL", component = "N")].rolling(time = 3, center = True).mean().data

running_mean

running_mean.dropna("time")



for s in N110.station:
    for c in N110.component:
        running_mean = N110.measurements.loc[dict(station = "NAL", component = "N")].rolling(time = window_size, center = True).mean().data
        data.measurements.loc[dict(station = s, component = c)] -= running_mean



def meansubtract(data, window_size = 200):
    tempdata = data.copy(deep = True)
    for s in data.station:
        for c in data.component:
            running_mean = data.measurements.loc[dict(station = s, component = c)].rolling(time = window_size, center = True).mean().data
            tempdata.measurements.loc[dict(station = s, component = c)] -= running_mean

    return tempdata


N110_ms.time
N110.time



N110_ms = meansubtract(N110, 100)
plt.figure(figsize = (20, 8))
plt.plot(N110_ms.measurements.loc[dict(station = "NAL", time = N110.time[range(100, 1100)])])

plt.figure(figsize = (20, 8))
plt.plot(N110.measurements.loc[dict(station = "NAL", time = N110.time[range(100, 1100)])])







fig3 = plt.figure(figsize = (20, 30))
nplots = len(data2_ms.station) + 2
pcm3 = plt.subplot(nplots, 1, 1)
osp3 = plt.subplot(nplots, 1, 2) #order sub plot

pcm3.pcolormesh(pcf2)
# pcm.xlabel("time", fontsize = 20)
# pcm.ylabel("r", fontsize = 20)
# pcm.colorbar()
osp3.plot(order2[100:619])

for i in range(nplots-2):
    s = data2_ms.station[i]
    ax = plt.subplot(nplots, 1, i+3, sharex = pcm3)
    ax.plot(data2_ms.measurements.loc[dict(station = s, time = data2_ms.time[range(100, 619)])])
    ax.title.set_text(s.data)

fig3.savefig("aaa.png")




def pcf(data, dr = 0.3):
    #HIGHLY INEFFICIENT - TAKES AGES TO RUN. PROCEED WITH CAUTION
    normeddata = data.copy(deep = True)
    # dr = 0.3
    r_range = np.linspace(0, 2, 21)
    # t = data.time[5]
    N = len(data.station) #number of points
    results = np.zeros((len(r_range), len(data.time))) #pair correlation function

    for time_index in range(len(data.time)):
        t = data.time[time_index]
        for s in data.station:
            normeddata.measurements.loc[dict(station = s, time = t)] = vector_normalise(data.measurements.loc[dict(station = s, time = t)].data)

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0
            dists = np.zeros((N, N))

            for s1 in data.station:
                for s2 in data.station:
                    diff = data.measurements.loc[dict(station = s1, time = t)] - data.measurements.loc[dict(station = s2, time = t)]
                    dist = np.sqrt(np.nansum(diff**2))
                    if (max(0, r) < dist and dist < r+dr):
                        count += 1

            for

            results[r_index, time_index] = (r*count)/(3*dr*N**2)

    return results




def vector_normalise(v):
    return v/np.sqrt(sum(v**2))





def test(data, dr = 0.3):
    normeddata = data.copy(deep = True)
    r_range = np.linspace(0.1, 2, 21)
    N = len(data.station) #number of stations
    results = np.zeros((len(r_range), len(data.time))) #pair correlation function

    for time_index in range(len(data.time)):
        t = data.time[time_index]
        dists = np.zeros((N, N))

        for s in data.station:
            normeddata.measurements.loc[dict(station = s, time = t)] = vector_normalise(data.measurements.loc[dict(station = s, time = t)].data)

        for i in range(N):
            s1 = data.station[i]
            for j in range(i+1, N):
                s2 = data.station[j]

                diff = normeddata.measurements.loc[dict(station = s1, time = t)] - normeddata.measurements.loc[dict(station = s2, time = t)]
                dists[i, j] = np.sqrt(np.nansum(diff**2))

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0
            x1 = 1* (r < dists) #ones where distances greater than lower bound
            x2 = 1*  (dists < r+dr) #ones where distances less than upper bound

            for i in range(N): #basically counts places where they are both ones
                for j in range(i+1, N):
                    count += (x1[i, j] == x2[i, j])

            results[r_index, time_index] = (count)/(3*dr*(N*r)**2)

    return results




data_ms = meansubtract(data)

trent = test(data_ms)
ogtrent = test(data)
trent_unnormed = test_unnormed(data_ms)
trent_order = saf.order_params(data_ms)

svl.plot_mag_data(data_ms)

plt.figure(figsize = (20, 8))
plt.pcolormesh(trent)
plt.figure(figsize = (20, 8))
plt.pcolormesh(ogtrent)

trent_unnormed = test_unnormed(data_ms, np.linspace(0, 100, 21))
plt.figure(figsize = (20, 8))
plt.pcolormesh(trent_unnormed)


stackedplot(data_ms, trent, trent_order)







N110_ms = meansubtract(N110)

pcf_N110 = test(N110_ms)
plt.figure(figsize = (30, 8))
plt.pcolormesh(pcf_N110)

order_N110 = saf.order_params(N110_ms)

stackedplot(N110_ms, pcf_N110, order_N110, "N110")




























def test_unnormed(data, r_range = np.linspace(0, 200, 21)):
    normeddata = data.copy(deep = True)
    # r_range = np.linspace(0, 200, 21)
    dr = r_range[2] - r_range[1]
    N = len(data.station) #number of stations
    results = np.zeros((len(r_range), len(data.time))) #pair correlation function

    for time_index in range(len(data.time)):
        t = data.time[time_index]
        dists = np.zeros((N, N))

        # for s in data.station:
            # normeddata.measurements.loc[dict(station = s, time = t)] = vector_normalise(data.measurements.loc[dict(station = s, time = t)].data)

        for i in range(N):
            s1 = data.station[i]
            for j in range(i+1, N):
                s2 = data.station[j]

                diff = normeddata.measurements.loc[dict(station = s1, time = t)] - normeddata.measurements.loc[dict(station = s2, time = t)]
                dists[i, j] = np.sqrt(np.nansum(diff**2))

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0
            x1 = 1* (r < dists) #ones where distances greater than lower bound
            x2 = 1*  (dists < r+dr) #ones where distances less than upper bound

            for i in range(N): #basically counts places where they are both ones
                for j in range(i+1, N):
                    count += (x1[i, j] == x2[i, j])

            results[r_index, time_index] = (r*count)/(3*dr*N**2)

    return results







def stackedplot(data, pcf, order):
    nplots = len(data.station)+ 2

    fig = plt.figure(figsize = (40, 60))

    pcm = plt.subplot(nplots, 1, 1)
    osp = plt.subplot(nplots, 1, 2) #order sub plot

    pcm.pcolormesh(pcf)
    # pcm.set_xlabel("time", fontsize = 20)
    pcm.set_ylabel("r, dr = " + "%s" %0.3, fontsize = 20)
    # pcm.colorbar()
    osp.plot(order)

    for i in range(nplots-2):
        s = data.station[i]
        ax = plt.subplot(nplots, 1, i+3, sharex = pcm)
        ax.plot(data.measurements.loc[dict(station = s)])
        ax.title.set_text(s.data)

    filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.savefig("%s.png" %filename)






def everything(data, r_range = np.linspace(0, 2, 21), dr = 0.3, normalised = 1):
    #first calculates pcf
    normeddata = data.copy(deep = True)
    if dr == 0:
        dr = r_range[2] - r_range[1]
    N = len(data.station) #number of stations
    pcf = np.zeros((len(r_range), len(data.time))) #pair correlation function

    for time_index in range(len(data.time)):
        t = data.time[time_index]
        dists = np.zeros((N, N))

        if normalised:
            for s in data.station:
                normeddata.measurements.loc[dict(station = s, time = t)] = vector_normalise(data.measurements.loc[dict(station = s, time = t)].data)

        for i in range(N):
            s1 = data.station[i]
            for j in range(i+1, N):
                s2 = data.station[j]

                diff = normeddata.measurements.loc[dict(station = s1, time = t)] - normeddata.measurements.loc[dict(station = s2, time = t)]
                dists[i, j] = np.sqrt(np.nansum(diff**2))

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0
            x1 = 1* (r < dists) #ones where distances greater than lower bound
            x2 = 1*  (dists < r+dr) #ones where distances less than upper bound

            for i in range(N): #basically counts places where they are both ones
                for j in range(i+1, N):
                    count += (x1[i, j] == x2[i, j])

            pcf[r_index, time_index] = (r*count)/(3*dr*N**2)

    #get order parameters
    order = saf.order_params(data)


    #plotting bit
    nplots = len(data.station)+ 2

    fig = plt.figure(figsize = (40, 60))

    pcm = plt.subplot(nplots, 1, 1)
    osp = plt.subplot(nplots, 1, 2) #order sub plot

    pcm.pcolormesh(pcf)
    # pcm.set_xlabel("time", fontsize = 20)
    pcm.ylabel("r, dr = " + "%s" %dr, fontsize = 20)
    # pcm.colorbar()
    osp.plot(order)

    for i in range(nplots-2):
        s = data.station[i]
        ax = plt.subplot(nplots, 1, i+3, sharex = pcm)
        ax.plot(data.measurements.loc[dict(station = s)])
        ax.title.set_text(s.data)

    filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.savefig("%s.png" %filename)
