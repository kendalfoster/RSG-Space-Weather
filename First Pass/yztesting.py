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
auto colour range based on max/min lat

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
from PIL import Image
import pandas as pd

station_readings = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

t = station_readings.time[1]
t.data
list_of_stations = station_readings.station






def cca(ds, components=['N', 'E', 'Z']):
    """
    Run canonical correlation analysis between stations.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    components : list, optional
        List of components in the data. Default is ['N', 'E', 'Z'].

    Returns
    -------
    xarray.Dataset
        Dataset containing the canonical correlation analysis attributes.
            The data_vars are: coeffs, weights, angles, comps.\n
            The coordinates are: first_st, second_st, component, index, ab, uv.
    """

    # detrend input Dataset, remove NAs
    # ds = mag_csv_to_Dataset(csv_file = "First Pass/dik1996.csv")
    ds = mag_detrend(ds)
    ds = ds.dropna(dim = 'time')

    # universal constants
    stations = ds.station.values
    num_st = len(stations)
    num_ws = len(components)
    num_cp = len(ds.time)

    # setup (symmetric) arrays for each attribute
    coeffs_arr = np.zeros(shape = (num_st, num_st), dtype = float)
    weights_arr = np.zeros(shape = (num_st, num_st, 2, num_ws), dtype = float)
    angles_arr = np.zeros(shape = (num_st, num_st), dtype = float)
    comps_arr = np.zeros(shape = (num_st, num_st, 2, num_cp), dtype = float)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        st_1 = ds.measurements.loc[dict(station = stations[i])]
        for j in range(i+1, num_st):
            st_2 = ds.measurements.loc[dict(station = stations[j])]
            # remove NaNs from data (will mess up cca)
            comb_st = xr.concat([st_1, st_2], dim = 'component')
            comb_st = comb_st.dropna(dim = 'time', how = 'any')
            st_1 = comb_st[:, 0:num_ws]
            st_2 = comb_st[:, num_ws:2*num_ws]
            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([st_1, st_2])
            ## store cca attributes ##
            # coeffs
            coeffs_arr[i,j] = ccac.cancorrs[0]
            coeffs_arr[j,i] = coeffs_arr[i,j] # mirror results
            # weights
            w0 = ccac.ws[0].flatten()
            w1 = ccac.ws[1].flatten()
            weights_arr[i,j,0,:] = w0
            weights_arr[i,j,1,:] = w1
            weights_arr[j,i,0,:] = w0 # mirror results
            weights_arr[j,i,1,:] = w1 # mirror results
            # angles
            angles_arr[i,j] = np.rad2deg(np.arccos(np.clip(np.dot(w0, w1), -1.0, 1.0)))
            angles_arr[j,i] = angles_arr[i,j] # mirror results
            # comps
            comps_arr[i,j,0,:] = ccac.comps[0].flatten()
            comps_arr[i,j,1,:] = ccac.comps[1].flatten()
            comps_arr[j,i,0,:] = comps_arr[i,j,0,:]
            comps_arr[j,i,1,:] = comps_arr[i,j,1,:]

    # build Dataset from coeffs
    coeffs = xr.Dataset(data_vars = {'coeffs': (['first_st', 'second_st'], coeffs_arr)},
                        coords = {'first_st': stations,
                                  'second_st': stations})
    # build Dataset from weights
    weights = xr.Dataset(data_vars = {'weights': (['first_st', 'second_st', 'ab', 'component'], weights_arr)},
                         coords = {'first_st': stations,
                                   'second_st': stations,
                                   'ab': ['a', 'b'],
                                   'component': components})
    # build Dataset from angles
    angles = xr.Dataset(data_vars = {'angles': (['first_st', 'second_st'], angles_arr)},
                        coords = {'first_st': stations,
                                  'second_st': stations})
    # build Dataset from comps
    comps = xr.Dataset(data_vars = {'comps': (['first_st', 'second_st', 'uv', 'index'], comps_arr)},
                       coords = {'first_st': stations,
                                 'second_st': stations,
                                 'uv': ['u', 'v'],
                                 'index': range(num_cp)})

    # merge Datasets
    res = xr.merge([coeffs, weights, angles, comps])

    return res



sm.cca(station_readings)
