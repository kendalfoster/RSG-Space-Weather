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


t = N330.time[1].data
t


N330.measurements.loc[dict(component = "N", time = t)].data

def vector_normalise(v):
    return v/np.sqrt(sum(v**2))

n = len(N330.station)

sum_N = 0

for s in data.station:
    if





sva.data_globe_gif(original, time_end = 20)


np.nansum(N330.measurements.loc[dict(component = "N", time = data.time[800])])





sum_E = sum(N330.measurements.loc[dict(component = "E", time = t)])
sum_Z = sum(N330.measurements.loc[dict(component = "Z", time = t)])

normed_sum = np.sqrt(sum_N**2 + sum_E**2 + sum_Z**2)
normed_sum

# np.sqrt(sum(sum(N330.measurements.loc[dict(station = s, time = t)] for s in N330.station)**2))




v_0 = sum(np.sqrt(sum(N330.measurements.loc[dict(station = s, component = c, time = t)]**2 for c in N330.component)) for s in N330.station)/n
#average absolute velocity

phi = (normed_sum/(n*v_0)).data.item()

phi

def order_param(data, normed = False):
    #there might be some difference if i normed each reading before doing the calculations, as some stations are more
    #sensitive than others
    #need to look into dealing with missing readings - currently looks like
    #if readings missing from one station then everything is fucked. should be quite simple.

    n = len(data.station)
    params = np.zeros(len(data.time))
    i = 0

    for t in data.time:
        sum_N = np.nansum(data.measurements.loc[dict(component = "N", time = t)])
        sum_E = np.nansum(data.measurements.loc[dict(component = "E", time = t)])
        sum_Z = np.nansum(data.measurements.loc[dict(component = "Z", time = t)])
        normed_sum = np.sqrt(sum_N**2 + sum_E**2 + sum_Z**2)
        # np.sqrt(sum(sum(N330.measurements.loc[dict(station = s, time = t)] for s in N330.station)**2))

        v_0 = np.sum(np.sqrt(np.nansum(data.measurements.loc[dict(station = s, time = t)]**2)) for s in data.station)/n
        #average absolute velocity

        phi = (normed_sum/(n*v_0))#.data.item()
        #calculates actual order parameter
        params[i] = phi

        i += 1

    return params

plt.plot(order_param(data))

params = np.zeros(len(times))
i = 0
for t in times:
    params[i] = order_param(data, t)
    i += 1


i
times.data


def order_param_plot(data):
    times = data.time
    params = np.zeros(len(times))
    i = 0

    for t in times:
        params[i] = order_param(data, t)
        i += 1

    plt.figure(figsize = (20, 8))
    plt.plot(params)
    plt.title("order parameter new")
    plt.xlabel("time")
    plt.ylabel("phi")

order_param_plot(data)

range(len(data.time))

a = data.measurements.loc[dict(station = "BLC", time = t)]

b = data.measurements.loc[dict(station = data.station[2], time = t)]


a-b

test = a-b
np.sqrt(np.nansum(test**2))





def pcf(data, dr = 0.3):
    # dr = 0.3
    r_range = np.linspace(0, 2, 21)
    # t = data.time[5]
    N = len(data.station) #number of points
    results = np.zeros((len(r_range), len(data.time[0:500]))) #pair correlation function

    for time_index in range(len(data.time[0:500])):
        t = data.time[time_index]
        for s in data.station:
            data.measurements.loc[dict(station = s, time = t)] = vector_normalise(data.measurements.loc[dict(station = s, time = t)].data)

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0

            for s1 in data.station:
                for s2 in data.station:
                    diff = data.measurements.loc[dict(station = s1, time = t)] - data.measurements.loc[dict(station = s2, time = t)]
                    dist = np.sqrt(np.nansum(diff**2))
                    if (max(0, r) < dist and dist < r+dr):
                        count += 1

            results[r_index, time_index] = (r*count)/(3*dr*N**2)

    return results



allpcf = pcf(original.loc[dict(time = original.time[:500])])


plt.figure(figsize = (24.5, 8))
plt.plot(params[0:500])
plt.title("order parameter new")
plt.xlabel("time")
# plt.xticks(data.time.data)
plt.ylabel("phi")


plt.figure(figsize = (27, 8))
plt.pcolormesh(pcf)
plt.xlabel("time", fontsize = 20)
plt.ylabel("r", fontsize = 20)
plt.colorbar()





plt.figure(figsize = (27, 8))
plt.pcolormesh(first6pcf)
plt.xlabel("time", fontsize = 20)
plt.ylabel("r", fontsize = 20)
plt.colorbar()

original.measurements.loc[dict(time = original.time[range(500)])].plot.line(x='time', hue='component', col='station', figsize = (20, 20), col_wrap=1)


order = order_param(original.loc[dict(time = original.time[range(500)])])

order





fig = plt.figure(figsize = (20, 30))
nplots = len(original.station) + 2
pcm = plt.subplot(nplots, 1, 1)
osp = plt.subplot(nplots, 1, 2) #order sub plot

pcm.pcolormesh(first6pcf)
# pcm.xlabel("time", fontsize = 20)
# pcm.ylabel("r", fontsize = 20)
# pcm.colorbar()
osp.plot(order[:500])

for i in range(nplots-2):
    s = data.station[i]
    ax = plt.subplot(nplots, 1, i+3, sharex = pcm)
    ax.plot(original.measurements.loc[dict(time = original.time[range(500)], station = s)])
    ax.title.set_text(s.data)

fig.savefig("test.png")






tal = plt.subplot(312)
blc = plt.subplot(313, sharex = tal)


tal.plot(original.measurements.loc[dict(time = original.time[range(500)], station = "TAL")])
tal.title.set_text("TAL")
blc.plot(original.measurements.loc[dict(time = original.time[range(500)], station = "BLC")])
blc.title.set_text("BLC")



tal.get_shared_x_axes().join(tal, blc, pcm)
tal.set_xticklabels([])

plt.plot(original.measurements.loc[dict(time = original.time[range(500)], station = "BLC")])
plt.xticks()



data.station[:6]


plt.figure(figsize = (20, 8))
plt.plot(pcf)
# plt.xticks(r_range)




vector_normalise(data.measurements.loc[dict(station = "BLC", time = t)].data)




first6stns = data.loc[dict(station = data.station[:6])]
first6pcf = pcf(first6stns)


data.measurements.loc[dict(station = data.station[:6])].loc[dict(time = t)]

data.measurements.loc[dict(time = data.time[range(100)])]













plt.figure(figsize = (24.5, 8))
plt.plot(params)
plt.title("order parameter new")
plt.xlabel("time")
# plt.xticks(data.time.data)
plt.ylabel("phi")

plt.figure(figsize = (27, 8))
plt.pcolormesh(pcf)
plt.xlabel("time", fontsize = 20)
plt.ylabel("r", fontsize = 20)
plt.colorbar()

data.measurements.loc[dict(time = data.time[range(100)])].plot.line(x='time', hue='station', col='component', figsize = (20, 8), col_wrap=1)
















avg_N = sum(data.measurements.loc[dict(component = "N", time = t)])/n
avg_E = sum(data.measurements.loc[dict(component = "E", time = t)])/n
avg_Z = sum(data.measurements.loc[dict(component = "Z", time = t)])/n

np.asarray((avg_N.data.item(), avg_E.data.item(), avg_Z.data.item()))



avg
np.dot(avg, data.measurements.loc[dict(station = "BLC", time = t)])

np.sqrt(sum(np.square(data.measurements.loc[(dict(station = "BLC", time = t))].data)))



n = len(data)
diffs = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

for t in data.time:
    avg_N = sum(data.measurements.loc[dict(component = "N", time = t)])/n
    avg_E = sum(data.measurements.loc[dict(component = "E", time = t)])/n
    avg_Z = sum(data.measurements.loc[dict(component = "Z", time = t)])/n

    avg = np.asarray((avg_N.data.item(), avg_E.data.item(), avg_Z.data.item()))
    abs_avg = np.sqrt(sum(np.square(avg)))

    for s in data.station:
        dot = np.dot(avg, data.measurements.loc[dict(station = s, time = t)])
        abs_st = np.sqrt(sum(np.square(data.measurements.loc[dict(station = "BLC", time = t)].data)))
        #magnitude of the reading at station s time t

        diff = np.rad2deg(np.arccos(dot/(abs_avg*abs_st)))
        diffs.measurements.loc[dict(station = s, time = t)] = diff



diffs









for t in data.time:
    for s in data.station:
        diffs.measurements.loc[dict(station = s, time = t)] -= avg

diffs
