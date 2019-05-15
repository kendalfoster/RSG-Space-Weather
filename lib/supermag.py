## Packages
import numpy as np
from numpy import hstack
import scipy as sp
from scipy import signal
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
# Our Packages
import lib.rcca as rcca

# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html

## Unused Packages, but potentially useful
# import scipy.signal as scg
# import xscale.signal.fitting as xsf # useful functions for xarray data structures
    # pip3 install git+https://github.com/serazing/xscale.git
    # pip3 install toolz



################################################################################
####################### Restructure ############################################
### Function to restructure the SuperMAG data as an xarray.Dataset
#       inputs: csv_file- SuperMAG data as csv file, downloaded from SuperMAG website
#               components- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLT- input False if the magnetic local time column is NOT included, default is True
#               MLAT- input False if the magnetic latitude column is NOT included, default is True
#       output: Dataset with the SuperMAG data easily accessible,
#                   data_vars- measurements, mlts, mlats
#                   coordinates- time, component, station
#                 ----time is first dimension (ie, axis=0 for numpy commands)
#                 ----data in DataArray format via output.measurements
#                 ----data in array format via output.measurements.values
def mag_csv_to_Dataset(csv_file, components=['N', 'E', 'Z'], MLT=True, MLAT=True):
    # get universally needed things
    data = pd.read_csv(csv_file)
    times = pd.to_datetime(data['Date_UTC'].unique())

    #-----------------------------------------------------------------------
    #---------- optional arguments -----------------------------------------
    #-----------------------------------------------------------------------

    # if MLAT is included, sort and make Dataset
    if MLAT is True:
        # sort stations by magnetic latitude (from north to south)
        stations = data['IAGA'].unique()
        num_st = len(stations)
        mlat_arr = np.vstack((stations,np.zeros(num_st))).transpose()
        for i in range(0,num_st):
            mlat_arr[i,1] = data['MLAT'].loc[data['IAGA'] == stations[i]].mean()
        mlat_arr = sorted(mlat_arr, key=lambda x: x[1], reverse=True)
        stations = [i[0] for i in mlat_arr]
        mlats = [round(i[1],4) for i in mlat_arr]
        # build MLAT Dataset, for merging later
        ds_mlat = xr.Dataset(data_vars = {'mlats': (['station'], mlats)},
                             coords = {'station': stations})
    elif MLAT is not True:
        stations = data['IAGA'].unique()

    # if MLT (Magnetic Local Time) is included, make a Dataset
    if MLT is True:
        # initialize DataArray (so we can append things to it later)
        cols_mlt = ['Date_UTC', 'MLT']
        temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == stations[0]]
        temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
        mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                           coords = [temp_times_mlt],
                           dims = ['time'])
        # loop through the stations and append each to master DataArray
        for i in stations[1:]:
            temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == i]
            temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
            temp_mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                                    coords = [temp_times_mlt],
                                    dims = ['time'])
            mlt = xr.concat([mlt, temp_mlt], dim = 'station')
            mlt = mlt.transpose('time', 'station')
        # build MLT Dataset, for merging later
        ds_mlt = xr.Dataset(data_vars = {'mlts': (['time', 'station'], mlt)},
                            coords = {'time': times,
                                      'station': stations})



    #-----------------------------------------------------------------------
    #---------- build the main DataArray of the measurements ---------------
    #-----------------------------------------------------------------------

    # initialize DataArray (so we can append things to it later)
    cols = np.append('Date_UTC', components)
    temp_data = data[cols].loc[data['IAGA'] == stations[0]]
    temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
    da = xr.DataArray(data = temp_data[components],
                      coords = [temp_times, components],
                      dims = ['time', 'component'])

    # loop through the stations and append each to master DataArray
    for i in stations[1:]:
        temp_data = data[cols].loc[data['IAGA'] == i]
        temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
        temp_da = xr.DataArray(data = temp_data[components],
                               coords = [temp_times, components],
                               dims = ['time', 'component'])
        da = xr.concat([da, temp_da], dim = 'station')
        da = da.transpose('time', 'component', 'station')

    # build Dataset from components
    ds = xr.Dataset(data_vars = {'measurements': (['time', 'component', 'station'], da)},
                    coords = {'time': times,
                              'component': components,
                              'station': stations})



    #-----------------------------------------------------------------------
    #---------- build the final DataArray from optional arguments ----------
    #-----------------------------------------------------------------------

    # include MLT
    if MLT is True:
        ds = xr.merge([ds, ds_mlt])

    # include MLAT
    if MLAT is True:
        ds = xr.merge([ds, ds_mlat])


    return ds
################################################################################




################################################################################
####################### Plotting ###############################################
### Function to plot the components like on the SuperMAG website
#       input: ds- dataset output from mag_csv_to_Dataset function
#       output: series of plots, one per station, of the components
def plot_mag_data(ds):
    ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)
################################################################################




################################################################################
####################### Detrending #############################################
### Function to detrend the data over time
#       input: ds- dataset output from mag_csv_to_Dataset function
#              type- type of detrending passed to scipy detrend, default linear
#       output: Dataset with measurements detrended
#                   data_vars- measurements
#                   coordinates- time, component, station
def mag_detrend(ds, type='linear'):
    stations = ds.station
    components = ds.component

    # initialize DataArray
    temp = ds.measurements.loc[dict(station = stations[0])]
    temp = temp.dropna(dim = 'time', how = 'any')
    temp_times = temp.time
    det = scg.detrend(data=temp, axis=0, type=type)
    da = xr.DataArray(data = det,
                      coords = [temp_times, components],
                      dims = ['time', 'component'])

    for i in range(1, len(stations)):
        temp = ds.measurements.loc[dict(station = stations[i])]
        temp = temp.dropna(dim = 'time', how = 'any')
        temp_times = temp.time
        det = scg.detrend(data=temp, axis=0, type=type)
        temp_da = xr.DataArray(data = det,
                               coords = [temp_times, components],
                               dims = ['time', 'component'])
        da = xr.concat([da, temp_da], dim = 'station')

    # fix coordinates
    da = da.assign_coords(station = stations)

    # convert DataArray into Dataset
    res = da.to_dataset(name = 'measurements')

    return res
################################################################################




################################################################################
####################### Windowing ##############################################
### Function to window the data at a given window length
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#       output: Dataset with extra dimension from windowing
#                   data_vars- measurements, mlts, mlats
#                   coordinates- time, component, station, window
def window(ds, win_len=128):
    # create a rolling object
    ds_roll = ds.rolling(time=win_len).construct(window_dim='win_rel_time').dropna('time')
    # fix window coordinates
    ds_roll = ds_roll.assign_coords(win_rel_time = range(win_len))
    ds_roll = ds_roll.rename(dict(time = 'win_start'))
    # ensure the coordinates are in the proper order
    ds_roll = ds_roll.transpose('win_start', 'component', 'station', 'win_rel_time')

    return ds_roll
################################################################################




################################################################################
####################### Canonical Correlation Analysis #########################
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


def cca_coeffs(ds, components=['N', 'E', 'Z']):
    """
    Calculate the first canonical correlation coefficients between stations.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    components : list, optional
        List of components in the data. Default is ['N', 'E', 'Z'].

    Returns
    -------
    xarray.Dataset
        Dataset containing the first canonical correlation coefficients.
            The data_vars are: cca_coeffs.\n
            The coordinates are: first_st, second_st.
    """

    # universal constants
    stations = ds.station
    num_st = len(stations)
    num_comp = len(components)

    # setup (triangular) array for the correlation coefficients
    cca_coeffs = np.zeros(shape = (num_st, num_st), dtype = float)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        first_st = ds.measurements.loc[dict(station = stations[i])]
        for j in range(i+1, num_st):
            second_st = ds.measurements.loc[dict(station = stations[j])]
            # remove NaNs from data (will mess up cca)
            comb_st = xr.concat([first_st, second_st], dim = 'component')
            comb_st = comb_st.dropna(dim = 'time', how = 'any')
            first_st = comb_st[:, 0:num_comp]
            second_st = comb_st[:, num_comp:2*num_comp]
            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([first_st, second_st])
            cca_coeffs[i,j] = ccac.cancorrs[0]
            cca_coeffs[j,i] = ccac.cancorrs[0]

    # build DataArray from the cca_coeffs array
    da = xr.DataArray(data = cca_coeffs,
                      coords = [stations, stations],
                      dims = ['first_st', 'second_st'])

    # convert the DataArray into a Dataset
    res = da.to_dataset(name = 'cca_coeffs')

    return res
################################################################################




################################################################################
####################### Thresholding ###########################################
### Function to calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st'
def mag_thresh_kf(ds, components=['N', 'E', 'Z']):
    thr = inter_st_cca(ds=ds, components=components)
    thr = thr.rename(dict(cca_coeffs = 'thresholds'))
    return thr


### Function to calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#              n0- the desired expected normalized degree of each node (station)
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st'
def mag_thresh_dods(ds, n0=0.25, components=['N', 'E', 'Z']):
    # univeral constants
    stations = ds.station.values
    num_st = len(stations)
    ct_mat = inter_st_cca(ds=ds, components=components)
    ct_vec = np.linspace(start=0, stop=1, num=101)

    # initialize
    arr = np.zeros(shape = (len(ct_vec), num_st))
    # iterate through all possible ct values
    for i in range(len(ct_vec)):
        temp = ct_mat.where(ct_mat > ct_vec[i], 0) # it looks opposite, but it's right
        temp = temp.where(temp <= ct_vec[i], 1)
        for j in range(num_st):
            arr[i,j] = sum(temp.loc[dict(first_st = stations[j])].cca_coeffs.values) + sum(temp.loc[dict(second_st = stations[j])].cca_coeffs.values)
    # normalize
    arr = arr/(num_st-1)

    # find indices roughly equal to n0 and get their values
    idx = np.zeros(num_st, dtype=int)
    thr = np.zeros(num_st)
    for i in range(num_st):
        idx[i] = int(np.where(arr[:,i] <= n0)[0][0])
        thr[i] = ct_vec[idx[i]]

    # create threshold matrix using smaller threshold in each pair
    threshold = np.zeros(shape = (num_st, num_st))
    for i in range(num_st):
        for j in range(i+1, num_st):
            if thr[i] < thr[j]:
                threshold[i,j] = thr[i]
                threshold[j,i] = thr[i]
            else:
                threshold[i,j] = thr[j]
                threshold[j,i] = thr[j]

    # restructure into Dataset
    res = xr.Dataset(data_vars = {'thresholds': (['first_st', 'second_st'], threshold)},
                     coords = {'first_st': stations,
                               'second_st': stations})

    return res


### Function to ultimately calculate the threshold for each station pair
#       input: ds- dataset output from mag_csv_to_Dataset function
#              win_len- length of the window, default is 128
#              n0- the desired expected normalized degree of each node (station)
#       output: Dataset of thresholds
#                   data- thresholds
#                   coordinates- 'first_st', 'second_st', 'win_start'
def threshold_ds(ds, win_len=128, n0=0.25, components=['N', 'E', 'Z']):
    # run window over data
    ds_win = sm.window(ds=ds, win_len=win_len)

    # format Dataset
    ds_win = ds_win.transpose('win_rel_time', 'component', 'station', 'win_start')
    ds_win = ds_win.rename(dict(win_rel_time = 'time'))

    # get threshold values for each window
    det = sm.mag_detrend(ds = ds_win[dict(win_start = 0)])
    net = sm.mag_thresh_dods(ds = det, n0=n0, components=components)
    for i in range(1, len(ds_win.win_start)):
        det = sm.mag_detrend(ds = ds_win[dict(win_start = i)])
        temp = sm.mag_thresh_dods(ds = det, n0=n0, components=components)
        net = xr.concat([net, temp], dim = 'win_start')

    # fix coordinates
    net = net.assign_coords(win_start = ds_win.win_start.values)

    return net

## Adjacency Matrix
def mag_adj_mat(ds, ds_win, n0=0.25, components=['N', 'E', 'Z']):
    stations = ds.station.values
    num_st = len(stations)

    cca = inter_st_cca(ds=ds_win, components= components)
    cca = cca.assign_coords(first_st = range(num_st))
    cca = cca.assign_coords(second_st = range(num_st))

    thresh = mag_thresh_dods(ds=ds, n0=n0, components=components)
    thresh = thresh.assign_coords(first_st = range(num_st))
    thresh = thresh.assign_coords(second_st = range(num_st))

    adj_mat = cca - thresh.thresholds
    values = adj_mat.cca_coeffs.values
    values[values > 0] = 1
    values[values <= 0] = 0
    adj_mat.cca_coeffs.values = values
    adj_mat = adj_mat.assign_coords(first_st = stations)
    adj_mat = adj_mat.assign_coords(second_st = stations)

    return adj_mat

def print_mag_adj_mat(ds, ds_win, n0=0.25, components=['N', 'E', 'Z']):
    stations = ds.station.values
    num_st = len(stations)

    cca = inter_st_cca(ds=ds_win, components= components)
    cca = cca.assign_coords(first_st = range(num_st))
    cca = cca.assign_coords(second_st = range(num_st))

    thresh = mag_thresh_dods(ds=ds, n0=n0, components=components)
    thresh = thresh.assign_coords(first_st = range(num_st))
    thresh = thresh.assign_coords(second_st = range(num_st))

    adj_mat = cca - thresh.thresholds
    values = adj_mat.cca_coeffs.values
    values[values > 0] = 1
    values[values <= 0] = 0
    adj_mat.cca_coeffs.values = values

    fig = plt.figure(figsize=(10,8))
    adj_mat.cca_coeffs.plot.pcolormesh(yincrease=False, cbar_kwargs={'label': 'CCA Threshold'})
    plt.title('Adjacency Matrix', fontsize=30)
    plt.xlabel('Station 1', fontsize=20)
    plt.xticks(ticks=range(num_st), labels=stations, rotation=0)
    plt.ylabel('Station 2', fontsize=20)
    plt.yticks(ticks=range(num_st), labels=stations, rotation=0)
    plt.show()

    return adj_mat
################################################################################




################################################################################
####################### Visualizing The Network ################################

##
def csv_to_coords():
    csv_file = "First Pass/20190420-12-15-supermag-stations.csv"
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


##
def data_globe_gif(station_components, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra"):
    #times in terms of index in the array, might be helpful to have a fn to look up index from timestamps
    names = []
    images = []
    list_of_stations = station_components.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = auto_ortho(list_of_stations)

    for i in range(time_start, time_end):
        t = station_components.time[i]
        fig = plot_data_globe(station_components, t, list_of_stations, ortho_trans)
        fig.savefig("gif/images_for_giffing/%s.png" %i)

    for i in range(time_start, time_end):
        names.append("gif/images_for_giffing/%s.png" %i)

    for n in names:
        frame = Image.open(n)
        images.append(frame)

    images[0].save("gif/%s.gif" %file_name, save_all = True, append_images = images[1:], duration = 50, loop = 0)


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




################################################################################
####################### Generating Model Data ##################################
# This function generates a time series (with no_of_components components) for a full day
# At some point during the day there will be Pc waves
# We input the start date and start time of the Pc waves; we generate data for the (full) start_date day
# The charateristics of the Pc waves are set by the inputs wavepacket_duration (in mins) and number_of_waves

# For generating underlying data we use an OU process
# Code comes from https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/

# phase_shift should be in the range [0,1]
# it specifies how much to offset the waves by, as a proportion of the full wave cycle

def generate_one_day_one_component_time_series(pc_wave_start_date, pc_wave_start_time, wavepacket_duration, number_of_waves, phase_shift = 0):

    date_time = pd.to_datetime(pc_wave_start_date + ' ' + pc_wave_start_time)
    total_timesteps = int(np.timedelta64(1,'D')/np.timedelta64(1,'m'))
    full_day_timeseries = np.zeros(total_timesteps)
    data_source = ['' for i in range(total_timesteps)]

    # first generate the wavepacket - a sine wave combined with a Gaussian window
    gaussian_window = signal.gaussian(wavepacket_duration+1, std=(wavepacket_duration+1)/6)
    sine_wave = np.zeros(wavepacket_duration+1)

    for minute in range(wavepacket_duration+1):
        sine_wave[minute] = np.sin((minute - phase_shift * wavepacket_duration/number_of_waves) * (2 * np.pi) * number_of_waves/wavepacket_duration)

    wavepacket_start_index = int((date_time-pd.to_datetime(pc_wave_start_date))/np.timedelta64(1,'m'))
    for i in range(wavepacket_duration+1):
        full_day_timeseries[wavepacket_start_index+i] = gaussian_window[i] * sine_wave[i] * 100
        data_source[wavepacket_start_index+i] = 'wavepacket'

    # next generate some random behaviour before and after the wavepacket
    # use an Ornstein-Uhlenbeck process (rewritten as a Langevin equation) to generate the other noisy data

    # first define the parameters
    # adjust sigma and tau to change the shape of the wavepacket
    sigma = 38  # Standard deviation. From (a single) empirical observation
    mu = 0.  # Mean.
    dt = 1.  # Time step.
    tau = 50. * dt # Time constant. This choice seems to yield reasonable-looking results
    T = 1440.  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.

    # things that are used in the formulae
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)

    # first complete the time series by populating the timesteps before the wavepacket
    # note that we use the time-reversibility property of the O-U process
    start_index_start = 0
    end_index_start = wavepacket_start_index

    # add 1 so that there is an overlap (of 1 timestep) between the OU process and the wavepacket
    # the first datapoint of the wavepacket will be used as the first datapoint of the O-U process
    first_part = np.zeros(end_index_start-start_index_start + 1)
    first_part[0] = full_day_timeseries[wavepacket_start_index]

    # populate the first part of the O-U process (before the wavepacket)
    for i in range(len(first_part) - 1):
        first_part[i + 1] = first_part[i] + dt * (-(first_part[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

    for i in range(len(first_part)):
        index = end_index_start - i
        full_day_timeseries[index] = first_part[i]
        if data_source[index] == 'wavepacket' and index != wavepacket_start_index:
            print('duplicate')
        elif data_source[index] == 'wavepacket' and index == wavepacket_start_index:
            data_source[index] = 'overlap'
        else:
            data_source[index] = 'OU_first_part'

    # now populate the last part of the O-U process (after the wavepacket)
    # note start_index_start, end_index_start, start_index_last and end_index_last are all array indices, hence the -1 in end_index_last
    start_index_last = wavepacket_start_index + wavepacket_duration
    end_index_last = int(np.timedelta64(1,'D')/np.timedelta64(1,'m')) - 1

    last_part = np.zeros(end_index_last - start_index_last + 1)
    last_part[0] = full_day_timeseries[start_index_last]

    # populate the last part of the O-U process (after the wavepacket)
    for i in range(len(last_part) - 1):
        last_part[i + 1] = last_part[i] + dt * (-(last_part[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

    for i in range(len(last_part)):
        index = start_index_last + i
        full_day_timeseries[index] = last_part[i]
        if (data_source[index] == 'wavepacket' or data_source[index] == 'OU_first_part') and index != start_index_last:
            print(index)
            print('duplicate')
        elif data_source[index] == 'wavepacket' and index == start_index_last:
            data_source[index] = 'overlap'
        else:
            data_source[index] = 'OU_last_part'

    return full_day_timeseries

def generate_one_day_time_series(pc_wave_start_date, pc_wave_start_time, wavepacket_duration, number_of_waves, phase_shift = [0, np.random.rand(), np.random.rand()], station = ['XXX']):
    day_start_time = pd.to_datetime(pc_wave_start_date)
    day_end_time = pd.to_datetime(pc_wave_start_date) + np.timedelta64(1,'D') - np.timedelta64(1,'m')
    total_timesteps = int(np.timedelta64(1,'D')/np.timedelta64(1,'m'))

    times = pd.to_datetime(np.linspace(day_start_time.value, day_end_time.value, total_timesteps))

    components = ['N', 'E', 'Z']
    measurement_data = np.zeros((len(times),len(components),len(station)))

    for station_index in range(len(station)):



        if station_index == 0:
            N_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:00:00', 30, 4, phase_shift = phase_shift[0])
            E_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:00:00', 30, 4, phase_shift = phase_shift[1])
            Z_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:00:00', 30, 4, phase_shift = phase_shift[2])
        else:
            N_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:06:00', 30, 4, phase_shift = phase_shift[0])
            E_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:06:00', 30, 4, phase_shift = phase_shift[1])
            Z_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:06:00', 30, 4, phase_shift = phase_shift[2])



        measurement_data[:,0,station_index] = N_component_time_series
        measurement_data[:,1,station_index] = E_component_time_series
        measurement_data[:,2,station_index] = Z_component_time_series

    dataarray = xr.DataArray(data = measurement_data,
                  coords = [times, components, station],
                  dims = ['time', 'component', 'station'])

    dataset = xr.Dataset(data_vars = {'measurements': (['time', 'component', 'station'], dataarray)},
                    coords = {'time': times,
                              'component': components,
                              'station': station})

    return dataset
################################################################################
