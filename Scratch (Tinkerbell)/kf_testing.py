## Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.gen_data as sag
import spaceweather.analysis.threshold as sat
import spaceweather.visualisation.animations as sva
import spaceweather.visualisation.static as svg
import spaceweather.visualisation.heatmaps as svh
import spaceweather.visualisation.lines as svl
import spaceweather.visualisation.spectral_analysis as svs
import spaceweather.supermag as sm
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ds1 = sad.csv_to_Dataset(csv_file="Data/20190403-00-22-supermag.csv", MLAT=True)
ds = ds1[dict(time = slice(30))]
X = ds[dict(station = 0)].measurements.values
Y = ds[dict(station = 1)].measurements.values

coeff, a, b = sac.cca(X, Y)

cca_ang = sac.cca_angles(ds)











    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        for j in range(i+1, num_st):
            lm = lag_mat_pair(ds = ds,
                              station1 = stations[i],
                              station2 = stations[j],
                              lag_range = lag_range,
                              win_len = win_len,
                              **kwargs)

            # append to master Dataset: dimension = second_st
            if j == i+1:
                lag_mat_ss = lm
            else:
                lag_mat_ss = xr.concat([lag_mat_ss, lm], dim = 'second_st')

        # append to master Dataset: dimension = first_st
        if i == 0:
            lag_mat = lag_mat_ss
        elif i < num_st-2:
            lag_mat = xr.concat([lag_mat, lag_mat_ss], dim = 'first_st')
        else: # if i = num_st-2
            dummy = lag_mat_ss.copy(deep = True)
            num_lag = len(lag_mat_ss.lag.values)
            num_win = len(lag_mat_ss.time_win.values)
            dummy.lag_coeffs.values = np.full(shape = (num_lag, num_win),
                                              fill_value = np.nan)
            dummy = dummy.assign_coords(first_st = stations[num_st-2],
                                        second_st = stations[num_st-2])
            lag_dum = xr.concat([lag_mat_ss, dummy], dim = 'second_st')
            lag_mat = xr.concat([lag_mat, lag_dum], dim = 'first_st')

    # finish the DataArrays
    time_wins = lag_mat_ss.time_win.values
    num_win = len(time_wins)
    lags = lag_mat_ss.lag.values
    num_lag = len(lags)
    win_sts = lag_mat_ss.win_start.values
    num_win_st = len(win_sts)
    lag_blank = np.full(shape=(num_win, num_lag, num_st, num_st, num_win_st), fill_value = np.nan)
    lag_bda = xr.Dataset(data_vars = {'lag_coeffs': (['time_win', 'lag', 'first_st', 'second_st', 'win_start'], lag_blank)},
                         coords = {'time_win': time_wins,
                                   'lag': lags,
                                   'first_st': stations,
                                   'second_st': stations,
                                   'win_start': win_sts})
    lag_mat = lag_mat.combine_first(lag_bda)

    # reorder the stations coordinates because xr.concat messed them up
    ds1 = lag_mat.loc[dict(first_st = stations[0])]
    for i in range(1, num_st):
        ds1 = xr.concat([ds1, lag_mat.loc[dict(first_st = stations[i])]], dim = 'first_st')
    ds2 = ds1.loc[dict(second_st = stations[0])]
    for i in range(1, num_st):
        ds2 = xr.concat([ds2, ds1.loc[dict(second_st = stations[i])]], dim = 'second_st')
    ds = ds2.transpose('time_win', 'lag', 'first_st', 'second_st', 'win_start')


































def cca_ds(ds, **kwargs):
    """
    Run canonical correlation analysis between stations and return an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.

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
            ds = sad.detrend(ds, **kwargs)


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
