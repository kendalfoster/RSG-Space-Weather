"""
Contents
--------

- cca
- cca_coeffs
- lag_mat
"""


## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca
import spaceweather.analysis.data_funcs as sad
import spaceweather.visualisation.heatmaps as svh



def cca(ds, detrend='linear'):
    """
    Run canonical correlation analysis between stations.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
    detrend : str or bool, optional
        Type of detrending to perform on ds prior to running canonical correlation
        analysis. If False is input, detrending will not occur. Default is linear.

    Returns
    -------
    xarray.Dataset
        Dataset containing the canonical correlation analysis attributes.
            The data_vars are: coeffs, weights, ang_rel, ang_abs, comps.\n
            The coordinates are: first_st, second_st, component, index, ab, uv.
    """

    #-------------------------------------------------------------------------------
    #---------------------- Preliminaries ------------------------------------------
    #-------------------------------------------------------------------------------
    # check if more than one station in Dataset
    stations = ds.station.values
    num_st = len(stations)
    if num_st <= 1:
        print('Error: only one station in Dataset')
        return 'Error: only one station in Dataset'

    # other constants
    components = ds.component.values
    num_ws = len(components)
    num_cp = len(ds.time)
    times = ds.time.values
    ab = ['a', 'b']
    uv = ['u', 'v']

    # detrend input Dataset
    if detrend is not False:
        if detrend is True:
            detrend = 'linear'
            ds = sad.detrend(ds, type = detrend)


    #---------------------------------------------------------------------------
    #---------------------- Loop through each station pair ---------------------
    #---------------------------------------------------------------------------
    for i in range(num_st-1): # first station
        ###----- initialize the DataArray -----###
        st_1 = ds.measurements.loc[dict(station = stations[i])]
        st_2 = ds.measurements.loc[dict(station = stations[i+1])]

        # remove NaNs from data (nans will mess up cca)
        both_st = xr.concat([st_1, st_2], dim = 'component')
        both_st = both_st.dropna(dim = 'time', how = 'any')

        # get constants
        temp_times = both_st.time.values
        ltt = len(temp_times)
        st_1 = both_st[:, 0:num_ws]
        st_2 = both_st[:, num_ws:2*num_ws]

        # run cca, suppress rcca output
        temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
        ccac = temp_cca.train([st_1, st_2])

        ###----- store cca attributes -----###
        # coefficients
        coeffs_arr = np.array([ccac.cancorrs[0].flatten()])
        coeffs_fs = xr.DataArray(data = coeffs_arr)
        coeffs_fs = coeffs_fs.assign_coords(first_st = i, second_st = i+1)
        coeffs_fs = coeffs_fs.squeeze(dim = ['dim_0', 'dim_1'], drop = True)

        # (cca) components
        comps_arr = np.array([ccac.comps[0].flatten(), ccac.comps[1].flatten()])
        comps_fs = xr.DataArray(data = comps_arr,
                                coords = [uv, temp_times],
                                dims = ['uv', 'time'])
        comps_fs = comps_fs.assign_coords(first_st = i, second_st = i+1)

        # weights
        w0 = ccac.ws[0].flatten()
        w1 = ccac.ws[1].flatten()
        weights_arr = np.array([w0, w1])
        weights_fs = xr.DataArray(data = weights_arr,
                                  coords = [ab, components],
                                  dims = ['ab', 'component'])
        weights_fs = weights_fs.assign_coords(first_st = i, second_st = i+1)

        # angles, weights relative to each other
        wt_norm = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(w1**2))
        ang_wts_arr = np.array([np.rad2deg(np.arccos(np.clip(np.dot(w0, w1)/wt_norm, -1.0, 1.0)))])
        ang_wts_fs = xr.DataArray(data = ang_wts_arr)
        ang_wts_fs = ang_wts_fs.assign_coords(first_st = i, second_st = i+1)
        ang_wts_fs = ang_wts_fs.squeeze(dim = ['dim_0'], drop = True)

        # angles, weights relative to measurements at each time
        ang_mes_arr = np.zeros(shape = (2,ltt))
        for k in range(ltt):
            xdata = st_1[dict(time=k)].values
            ydata = st_2[dict(time=k)].values
            wt_nrm0 = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(xdata**2))
            wt_nrm1 = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(ydata**2))
            ang_mes_arr[0,k] = np.rad2deg(np.arccos(np.clip(np.dot(w0, xdata)/wt_nrm0, -1.0, 1.0)))
            ang_mes_arr[1,k] = np.rad2deg(np.arccos(np.clip(np.dot(w1, ydata)/wt_nrm1, -1.0, 1.0)))
        ang_mes_fs = xr.DataArray(data = ang_mes_arr,
                                  coords = [ab, temp_times],
                                  dims = ['ab', 'time'])
        ang_mes_fs = ang_mes_fs.assign_coords(first_st = i, second_st = i+1)

        if num_st >= 3: # check if there are at least three stations
            ###----- loop through the other stations -----###
            for j in range(i+2, num_st): # second station in the pair
                st_2 = ds.measurements.loc[dict(station = stations[j])]

                # remove NaNs from data (nans will mess up cca)
                both_st = xr.concat([st_1, st_2], dim = 'component')
                both_st = both_st.dropna(dim = 'time', how = 'any')

                # get constants
                temp_times = both_st.time.values
                ltt = len(temp_times)
                st_1 = both_st[:, 0:num_ws]
                st_2 = both_st[:, num_ws:2*num_ws]

                # run cca, suppress rcca output
                temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
                ccac = temp_cca.train([st_1, st_2])

                ###----- store cca attributes, merge into initialized DataArray -----###
                # coefficients
                cfs_arr = np.array([ccac.cancorrs[0].flatten()])
                cfs = xr.DataArray(data = cfs_arr)
                cfs = cfs.assign_coords(first_st = i, second_st = j)
                cfs = cfs.squeeze(dim = ['dim_0', 'dim_1'], drop = True)
                coeffs_fs = xr.concat([coeffs_fs, cfs], dim = 'second_st')

                # (cca) components
                cps_arr = np.array([ccac.comps[0].flatten(), ccac.comps[1].flatten()])
                cps = xr.DataArray(data = cps_arr,
                                   coords = [uv, temp_times],
                                   dims = ['uv','time'])
                cps = cps.assign_coords(first_st = i, second_st = j)
                comps_fs = xr.concat([comps_fs, cps], dim = 'second_st')

                # weights
                w0 = ccac.ws[0].flatten()
                w1 = ccac.ws[1].flatten()
                wts_arr = np.array([w0, w1])
                wts = xr.DataArray(data = wts_arr,
                                   coords = [ab, components],
                                   dims = ['ab', 'component'])
                wts = wts.assign_coords(first_st = i, second_st = j)
                weights_fs = xr.concat([weights_fs, wts], dim = 'second_st')

                # angles, weights relative to each other
                wt_norm = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(w1**2))
                an_w_arr = np.array([np.rad2deg(np.arccos(np.clip(np.dot(w0, w1)/wt_norm, -1.0, 1.0)))])
                an_w = xr.DataArray(data = an_w_arr)
                an_w = an_w.assign_coords(first_st = i, second_st = j)
                an_w = an_w.squeeze(dim = ['dim_0'], drop = True)
                ang_wts_fs = xr.concat([ang_wts_fs, an_w], dim = 'second_st')

                # angles, weights relative to measurements at each time
                an_m_arr = np.zeros(shape = (2,ltt))
                for k in range(ltt):
                    xdata = st_1[dict(time=k)].values
                    ydata = st_2[dict(time=k)].values
                    wt_nrm0 = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(xdata**2))
                    wt_nrm1 = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(ydata**2))
                    an_m_arr[0,k] = np.rad2deg(np.arccos(np.clip(np.dot(w0, xdata)/wt_nrm0, -1.0, 1.0)))
                    an_m_arr[1,k] = np.rad2deg(np.arccos(np.clip(np.dot(w1, ydata)/wt_nrm1, -1.0, 1.0)))
                an_m = xr.DataArray(data = an_m_arr,
                                    coords = [ab, temp_times],
                                    dims = ['ab', 'time'])
                an_m = an_m.assign_coords(first_st = i, second_st = j)
                ang_mes_fs = xr.concat([ang_mes_fs, an_m], dim = 'second_st')


        ###----- merge above DataArrays into master DataAray -----###
        if i==0:
            coeffs = coeffs_fs
            comps = comps_fs
            weights = weights_fs
            ang_wts = ang_wts_fs
            ang_mes = ang_mes_fs
        elif i < num_st-2:
            coeffs = xr.concat([coeffs, coeffs_fs], dim = 'first_st')
            comps = xr.concat([comps, comps_fs], dim = 'first_st')
            weights = xr.concat([weights, weights_fs], dim = 'first_st')
            ang_wts = xr.concat([ang_wts, ang_wts_fs], dim = 'first_st')
            ang_mes = xr.concat([ang_mes, ang_mes_fs], dim = 'first_st')
        else: # i = num_st-2; ie the last loop
            # coefficients
            dummy = xr.DataArray(data = 0)
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            coeffs_dum = xr.concat([coeffs_fs, dummy], dim = 'second_st')
            coeffs = xr.concat([coeffs, coeffs_dum], dim = 'first_st')
            coeffs = coeffs.transpose('first_st', 'second_st')

            # (cca) components
            dum = np.zeros(shape = (2, ltt))
            dummy = xr.DataArray(data = dum,
                                 coords = [uv, temp_times],
                                 dims = ['uv','time'])
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            comps_dum = xr.concat([comps_fs, dummy], dim = 'second_st')
            comps = xr.concat([comps, comps_dum], dim = 'first_st')
            comps = comps.transpose('first_st', 'second_st', 'uv', 'time')

            # weights
            dum = np.zeros(shape = (2, num_ws))
            dummy = xr.DataArray(data = dum,
                                 coords = [ab, components],
                                 dims = ['ab','component'])
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            weights_dum = xr.concat([weights_fs, dummy], dim = 'second_st')
            weights = xr.concat([weights, weights_dum], dim = 'first_st')
            weights = weights.transpose('first_st', 'second_st', 'ab', 'component')

            # angles, weights relative to each other
            dummy = xr.DataArray(data = 0)
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            ang_wts_dum = xr.concat([ang_wts_fs, dummy], dim = 'second_st')
            ang_wts = xr.concat([ang_wts, ang_wts_dum], dim = 'first_st')
            ang_wts = ang_wts.transpose('first_st', 'second_st')

            # angles, weights relative to measurements at each time
            dum = np.zeros(shape = (2, ltt))
            dummy = xr.DataArray(data = dum,
                                 coords = [ab, temp_times],
                                 dims = ['ab','time'])
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            ang_mes_dum = xr.concat([ang_mes_fs, dummy], dim = 'second_st')
            ang_mes = xr.concat([ang_mes, ang_mes_dum], dim = 'first_st')
            ang_mes = ang_mes.transpose('first_st', 'second_st', 'ab', 'time')


    #---------------------------------------------------------------------------
    #---------------------- Finish the DataArrays ------------------------------
    #---------------------------------------------------------------------------
    ###----- adjust the coordinates -----###
    ns = range(num_st)

    # coefficients
    coeffs_blank = np.zeros(shape=(num_st, num_st))
    coeffs_bda = xr.DataArray(data = coeffs_blank,
                              coords = [ns, ns],
                              dims = ['first_st', 'second_st'])
    coeffs = coeffs.combine_first(coeffs_bda)

    # (cca) components
    comps_blank = np.zeros(shape=(num_st, num_st, 2, num_cp))
    comps_bda = xr.DataArray(data = comps_blank,
                             coords = [ns, ns, uv, times],
                             dims = ['first_st', 'second_st', 'uv', 'time'])
    comps = comps.combine_first(comps_bda)

    # weights
    weights_blank = np.zeros(shape=(num_st, num_st, 2, num_ws))
    weights_bda = xr.DataArray(data = weights_blank,
                             coords = [ns, ns, ab, components],
                             dims = ['first_st', 'second_st', 'ab', 'component'])
    weights = weights.combine_first(weights_bda)

    # angles, weights relative to each other
    ang_wts_blank = np.zeros(shape=(num_st, num_st))
    ang_wts_bda = xr.DataArray(data = ang_wts_blank,
                              coords = [ns, ns],
                              dims = ['first_st', 'second_st'])
    ang_wts = ang_wts.combine_first(ang_wts_bda)

    # angles, weights relative to measurements at each time
    ang_mes_blank = np.zeros(shape=(num_st, num_st, 2, num_cp))
    ang_mes_bda = xr.DataArray(data = ang_mes_blank,
                               coords = [ns, ns, ab, times],
                               dims = ['first_st', 'second_st', 'ab', 'time'])
    ang_mes = ang_mes.combine_first(ang_mes_bda)

    ###----- mirror results across the main diagonal -----###
    for i in range(num_st):
        for j in range(i+1, num_st):
            coeffs.values[j,i] = coeffs.values[i,j]
            comps.values[j,i,:,:] = comps.values[j,i,:,:]
            weights.values[j,i,:,:] = weights.values[j,i,:,:]
            ang_wts.values[j,i] = ang_wts.values[i,j]
            ang_mes.values[j,i,:,:] = ang_mes.values[i,j,:,:]

    ####----- label the station coordinates correctly -----###
    coeffs = coeffs.assign_coords(first_st = stations, second_st = stations)
    comps = comps.assign_coords(first_st = stations, second_st = stations)
    weights = weights.assign_coords(first_st = stations, second_st = stations)
    ang_wts = ang_wts.assign_coords(first_st = stations, second_st = stations)
    ang_mes = ang_mes.assign_coords(first_st = stations, second_st = stations)


    #-------------------------------------------------------------------------------
    #---------------------- Package the DataArrays as a Dataset --------------------
    #-------------------------------------------------------------------------------
    res = xr.Dataset(data_vars = {'coeffs' : coeffs,
                                  'comps'  : comps,
                                  'weights': weights,
                                  'ang_wts': ang_wts,
                                  'ang_mes': ang_mes})

    return res


def cca_coeffs(ds, **kwargs):
    """
    Calculate the first canonical correlation coefficients between stations.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.

    Returns
    -------
    xarray.Dataset
        Dataset containing the first canonical correlation coefficients.
            The data_vars are: cca_coeffs.\n
            The coordinates are: first_st, second_st.
    """

    # detrend input Dataset, remove NAs
    ds = sad.detrend(ds, **kwargs)
    ds = ds.dropna(dim = 'time')

    # universal constants
    stations = ds.station.values
    num_st = len(stations)
    components = ds.component.values
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


def lag_mat(ds, station1=None, station2=None, lag_range=10, win_len=128,
            plot=False, **kwargs):
    """
    Calculate a heatmap of correlations between two stations over time windows and lag.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.\n
        Note the time dimension must be at least win_len + 2*lag_range.
    station1 : str, optional
        Station for which the time window is fixed.
        If no station is provided, this will default to the first station in ds.
    station2 : str, optional
        Station for which the time window will be shifted according to lag_range.
        If no station is provided, this will default to the second station in ds.
    lag_range: int, optional
        The range, in minutes, of positive and negative shifts for station2.
        Default is 10.
    win_len : int, optional
        Length of window in minutes. Default is 128.
    plot : bool, optional
        Whether or not to plot and return the correlation matrix as a heatmap.
        Default is False.

    Returns
    -------
    xarray.Dataset
        Dataset containing the correlation coefficients.
            The data_vars are: cor_coeffs.\n
            The coordinates are: time_win, lag.
    matplotlib.figure.Figure
        Plot of the correlogram; ie heatmap of correlations.
    """

    # check if ds timeseries is long enough
    nt = len(ds.time.values)
    if nt < win_len + 2*lag_range:
        print('Error: ds timeseries < win_len + 2*lag_range')
        return 'Error: ds timeseries < win_len + 2*lag_range'

    # check if stations are provided
    stations = ds.station.values
    if len(stations) <= 1:
        print('Error: only one station in Dataset')
        return 'Error: only one station in Dataset'
    if station1 is None:
        station1 = stations[0]
    if station2 is None:
        station2 = stations[1]

    # Select the stations and window the data
    ds = ds.loc[dict(station = [station1,station2])]
    windowed = sad.window(ds, win_len)
    ts1 = windowed.loc[dict(station = station1)].measurements
    ts2 = windowed.loc[dict(station = station2)].measurements
    ts1 = ts1.transpose('win_len', 'component', 'win_start')
    ts2 = ts2.transpose('win_len', 'component', 'win_start')

    # Set up array
    time = range(lag_range+1, len(windowed.win_start)-lag_range+1)
    lag = range(-lag_range, lag_range+1)
    corr = np.zeros(shape = (len(lag), len(time)))

    # Calculate correlations
    for j in range(len(time)):
        for i in range(len(lag)):
            ts1_temp = ts1[dict(win_start = time[j]-1)]
            ts2_temp = ts2[dict(win_start = time[j]+lag[i]-1)]
            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([ts1_temp, ts2_temp])
            if ccac.cancorrs[0] < 0:
                print(i, j, "\n")
            corr[i,j] = ccac.cancorrs[0]

    lag_mat = xr.Dataset(data_vars = {'lag_coeffs': (['lag', 'time_win'], corr)},
                         coords = {'lag': lag,
                                   'time_win': time})

    # plot adjacency matrix
    if plot:
        fig = svh.plot_lag_mat(lag_mat = lag_mat, time_win = time, lag = lag)
        return lag_mat, fig
    else:
        return lag_mat
