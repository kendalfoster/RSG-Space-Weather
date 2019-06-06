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

def pcf(data, steps, dr = 0, normalised = 0):
    normeddata = data.copy(deep = True)
    if normalised:
        r_range = np.linspace(0, 3, 31)
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


def meansubtract(data, window_size = 200):
    tempdata = data.copy(deep = True)
    for s in data.station:
        for c in data.component:
            running_mean = data.measurements.loc[dict(station = s, component = c)].rolling(time = window_size, center = True).mean().data
            tempdata.measurements.loc[dict(station = s, component = c)] -= running_mean

    return tempdata


def vector_normalise(v):
    return v/np.sqrt(sum(v**2))


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

    # filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # fig.savefig(fname = "%s.pdf" %filename, dpi = "figure", format = "pdf")






quietday = sad.mag_csv_to_Dataset(csv_file = "Data/Report/quiet-day.csv", MLT = True, MLAT = True)
quietday_ms = meansubtract(quietday, 150)
quietday_ms.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1, figsize = (20, 40))

quiet_pcf_unnormed = pcf(quietday_ms, steps = 201)
quiet_pcf_normed = pcf(quietday_ms.loc[dict(time = quietday_ms.time[100:500])], steps = 201, normalised = 1)


quiet_order_param = saf.order_params(quietday_ms)

stackedplot(quietday_ms, quiet_pcf_normed, quiet_order_param)




storm = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
storm_ms = meansubtract(storm, 170)
storm_ms.measurements.loc[dict(time = storm_ms.time[300:650])].plot.line(x='time', hue='component', col='station', col_wrap=1, figsize = (10, 40))

storm_pcf_unnormed = pcf(storm_ms, steps = 201)
storm_pcf_normed = pcf(storm_ms.loc[dict(time = storm_ms.time[300:700])], steps = 201, normalised = 1)
storm_order_param = saf.order_params(storm_ms)

stackedplot(storm_ms, storm_pcf_unnormed, storm_order_param)


storm_pcf_unnormed.shape
# nplots = len(data.station)+ 2

# fig = plt.figure(figsize = (20, 30))

fig = plt.subplots(4, 2, figsize = (20, 12))
#stormy
pcm1 = plt.subplot(421)
# osp1 = plt.subplot(423, sharex = pcm1) #order sub plot

pcm1.pcolormesh(storm_pcf_unnormed[:, 300:700], cmap = "binary")
pcm1.set_title("Stormy Day", fontsize = 20)
pcm1.set_ylabel("r", fontsize = 15)
pcm1.set_xlim([0, 400])
# fig.colorbar(im, ax = pcm)
# osp1.plot(storm_order_param[300:700])
# osp1.set_ylim([0, 1])
# osp1.set_ylabel("φ", fontsize = 15)

stn1 = plt.subplot(425, sharex = pcm1)
stn1.plot(storm_ms.measurements.loc[dict(station = "RAN", time = storm_ms.time[300:700])])
stn1.axhline(y = 0, color = "k", linestyle = "--", lw = 0.5)
stn1.set_ylabel("RAN", fontsize = 15)

stn2 = plt.subplot(427, sharex = pcm1, sharey = stn1)
stn2.plot(storm_ms.measurements.loc[dict(station = "EKP", time = storm_ms.time[300:700])])
stn2.axhline(y = 0, color = "k", linestyle = "--", lw = 0.5)
stn2.set_xlabel("Timesteps (minutes)", fontsize = 15)
stn2.set_ylabel("EKP", fontsize = 15)


#quiet
pcm2 = plt.subplot(422, sharey = pcm1)
# osp2 = plt.subplot(424, sharex = pcm2, sharey = osp1) #order sub plot

pcm2.pcolormesh(quiet_pcf_unnormed[:, 100:500], cmap = "binary")
pcm2.set_title("Quiet Day", fontsize = 20)
# pcm1.set_xlabel("time", fontsize = 20)
# pcm2.set_ylabel("Radial distribution g(r)", fontsize = 20)
pcm2.set_xlim([0, 400])

pcmtest = plt.subplot(424, sharex = pcm2)
pcmtest.pcolormesh(quiet_pcf_normed)
pcmtest2 = plt.subplot(423, sharex = pcm1)
pcmtest2.pcolormesh(storm_pcf_normed)
# fig.colorbar(im, ax = pcm)
# osp2.plot(quiet_order_param[100:500])
# osp2.set_ylabel("Order parameter φ", fontsize = 20)

stn3 = plt.subplot(426, sharex = pcm2, sharey = stn1)
stn3.plot(quietday_ms.measurements.loc[dict(station = "RAN", time = quietday_ms.time[100:500])])
stn3.axhline(y = 0, color = "k", linestyle = "--", lw = 0.5)

stn4 = plt.subplot(428, sharex = pcm2, sharey = stn1)
stn4.plot(quietday_ms.measurements.loc[dict(station = "EKP", time = quietday_ms.time[100:500])])
stn4.axhline(y = 0, color = "k", linestyle = "--", lw = 0.5)
stn4.set_xlabel("Timesteps (minutes)", fontsize = 15)

filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fig[0].savefig(fname = "%s.pdf" %filename, dpi = "figure", format = "pdf")
