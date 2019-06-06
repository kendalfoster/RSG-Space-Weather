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


quietday = sad.mag_csv_to_Dataset(csv_file = "Data/Report/quiet-day.csv", MLT = True, MLAT = True)
quietday_ms = meansubtract(quietday, 150)
quietday_ms.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1, figsize = (20, 40))


everything(quietday_ms)





reportstorm = sad.mag_csv_to_Dataset(csv_file = "Data/Report/event-1997-01-06.csv", MLT = True, MLAT = True)
reportstorm_ms = meansubtract(reportstorm, 150)
reportstorm_ms.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1, figsize = (20, 40))


everything(reportstorm_ms)





test = sad.meansubtract(N110)

plt.figure(figsize = (20, 8))
plt.plot(test.measurements.loc[dict(station = "NAL", time = test.time[range(1500)])])




data_ms = meansubtract(data, 170)
data_ms.measurements.loc[dict(time = data_ms.time[300:650])].plot.line(x='time', hue='component', col='station', col_wrap=1, figsize = (10, 40))

everything(data_ms.loc[dict(time = data_ms.time[300:650])])









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


N110_ms = meansubtract(N110, 100)
plt.figure(figsize = (20, 8))
plt.plot(N110_ms.measurements.loc[dict(station = "NAL", time = N110.time[range(100, 1100)])])

plt.figure(figsize = (20, 8))
plt.plot(N110.measurements.loc[dict(station = "NAL", time = N110.time[range(100, 1100)])])








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




























def test_unnormed(data, r_range = np.linspace(1, 200, 21)):
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

        R = np.max(dists)/2 #radius of system???

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0
            x1 = 1* (r < dists) #ones where distances greater than lower bound
            x2 = 1*  (dists < r+dr) #ones where distances less than upper bound

            for i in range(N): #basically counts places where they are both ones
                for j in range(i+1, N):
                    count += (x1[i, j] == x2[i, j])

            results[r_index, time_index] = (R**3*count)/(3*dr*(N*r)**2)

    return results





def stackedplot(data, pcf, order):
    nplots = len(data.station)+ 2

    fig = plt.figure(figsize = (20, 30))

    pcm = plt.subplot(nplots, 1, 1)
    osp = plt.subplot(nplots, 1, 2, sharex = pcm) #order sub plot

    pcm.pcolormesh(pcf, cmap = "binary")
    pcm.set_xlabel("time", fontsize = 20)
    pcm.set_ylabel("r, dr = " + "%s" %0.3, fontsize = 20)
    pcm.set_xlim([100, len(order)-100])
    # fig.colorbar(im, ax = pcm)
    osp.plot(order)
    osp.set_ylabel("Order parameter φ", fontsize = 20)

    for i in range(nplots-2):
        s = data.station[i]
        ax = plt.subplot(nplots, 1, i+3, sharex = pcm)
        ax.plot(data.measurements.loc[dict(station = s)])
        ax.title.set_text(s.data)

    filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.savefig(fname = "%s.pdf" %filename, dpi = "figure", format = "pdf")





stackedplot(N110_ms, pcf_N110, order_N110)





def pcf_everything(data, steps, dr, normalised):
    normeddata = data.copy(deep = True)
    if normalised:
        r_range = np.linspace(0, 2, steps)
    else:
        r_range = np.linspace(0, 200, steps)

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

    return pcf


def everything(data, steps = 101, dr = 0, normalised = 0):
    #first calculates pcf
    pcf = pcf_everything(data, steps, dr, normalised)

    #get order parameters
    order = saf.order_params(data)

    stackedplot(data, pcf, order)

    # nplots = len(data.station)+ 2
    #
    # fig = plt.figure(figsize = (40, 60))
    #
    # pcm = plt.subplot(nplots, 1, 1)
    # osp = plt.subplot(nplots, 1, 2) #order sub plot
    #
    # pcm.pcolormesh(pcf, cmap = "binary")
    # pcm.set_xlabel("time", fontsize = 20)
    # pcm.set_ylabel("r, dr = " + "%s" %0.3, fontsize = 20)
    # # fig.colorbar(im, ax = pcm)
    # osp.plot(order)
    #
    # for i in range(nplots-2):
    #     s = data.station[i]
    #     ax = plt.subplot(nplots, 1, i+3, sharex = pcm)
    #     ax.plot(data.measurements.loc[dict(station = s)])
    #     ax.title.set_text(s.data)
    #
    # filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # fig.savefig(fname = "%s.pdf" %filename, dpi = "figure", format = "pdf")


, time = data_ms.time[range(200, 700)]
everything(data_ms.loc[dict(station = data_ms.station[range(6)])], steps = 101, dr = 4)
everything(data_ms.loc[dict(station = data_ms.station[range(6)])], normalised = 1)
