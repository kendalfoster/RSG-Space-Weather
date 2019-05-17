## Packages
import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca
import spaceweather.analysis.data_funcs as sad

## Need to fix length error that George spotted
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
            The data_vars are: coeffs, weights, ang_rel, ang_abs, comps.\n
            The coordinates are: first_st, second_st, component, index, ab, uv.
    """

    # detrend input Dataset
    ds = sad.detrend(ds)

    # universal constants
    stations = ds.station.values
    num_st = len(stations)
    num_ws = len(components)
    num_cp = len(ds.time)

    # setup (symmetric) arrays for each attribute
    coeffs_arr = np.zeros(shape = (num_st, num_st), dtype = float)
    weights_arr = np.zeros(shape = (num_st, num_st, 2, num_ws), dtype = float)
    ang_rel_arr = np.zeros(shape = (num_st, num_st), dtype = float)
    ang_abs_arr = np.zeros(shape = (num_st, num_st, num_cp, 2), dtype = float)
    comps_arr = np.zeros(shape = (num_st, num_st, 2, num_cp), dtype = float)

    for i in range(0, num_st-1):
        st_1 = ds.measurements.loc[dict(station = stations[i])]
        for j in range(i+1, num_st):
            st_2 = ds.measurements.loc[dict(station = stations[j])]
            # remove NaNs from data (no_nans = 0will mess up cca)
            comb_st = xr.concat([st_1, st_2], dim = 'component')
            comb_st_no_na = comb_st.dropna(dim = 'time', how = 'any')
            st_1 = comb_st[:, 0:num_ws]
            st_2 = comb_st[:, num_ws:2*num_ws]
            st_1_no_na = comb_st_no_na[:, 0:num_ws]
            st_2_no_na = comb_st_no_na[:, num_ws:2*num_ws]
            nan_times = []
            for scratch_time in comb_st.time.values:
                if np.isnan(sum(comb_st.loc[dict(time = scratch_time)].values)):
                    nan_times.append(scratch_time)

            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([st_1_no_na, st_2_no_na])
            ## store cca attributes ##
            # coeffs
            coeffs_arr[i,j] = ccac.cancorrs[0]
            coeffs_arr[j,i] = coeffs_arr[i,j] # mirror results
            # weights (a and b from Wikipedia)
            w0 = ccac.ws[0].flatten() # this is a
            w1 = ccac.ws[1].flatten() # this is b
            weights_arr[i,j,0,:] = w0
            weights_arr[i,j,1,:] = w1
            weights_arr[j,i,0,:] = w0 # mirror results
            weights_arr[j,i,1,:] = w1 # mirror results
            # angles, relative
            wt_norm = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(w1**2))
            ang_rel_arr[i,j] = np.rad2deg(np.arccos(np.clip(np.dot(w0, w1)/wt_norm, -1.0, 1.0)))
            ang_rel_arr[j,i] = ang_rel_arr[i,j] # mirror results
            # angles, absolute

            no_nans = 0
            for k,timestamp in enumerate(ds.time):
                if timestamp in nan_times:
                    ang_abs_arr[i,j,k,0] = np.nan
                    ang_abs_arr[i,j,k,1] = np.nan
                    ang_abs_arr[j,i,k,0] = np.nan
                    ang_abs_arr[j,i,k,1] = np.nan

                    comps_arr[i,j,0,k] = np.nan
                    comps_arr[i,j,1,k] = np.nan
                    comps_arr[j,i,0,k] = np.nan
                    comps_arr[j,i,1,k] = np.nan

                    no_nans += 1
                else:
                    xdata = st_1[dict(time=k)].values
                    ydata = st_2[dict(time=k)].values
                    wt_nrm0 = np.sqrt(np.sum(w0**2)) * np.sqrt(np.sum(xdata**2))
                    wt_nrm1 = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(ydata**2))
                    ang_abs_arr[i,j,k,0] = np.rad2deg(np.arccos(np.clip(np.dot(w0, xdata)/wt_nrm0, -1.0, 1.0)))
                    ang_abs_arr[i,j,k,1] = np.rad2deg(np.arccos(np.clip(np.dot(w1, ydata)/wt_nrm1, -1.0, 1.0)))
                    ang_abs_arr[j,i,k,0] = ang_abs_arr[i,j,k,0]
                    ang_abs_arr[j,i,k,1] = ang_abs_arr[i,j,k,1]

                    # comps (a^T*X and b^T*Y from Wikipedia)

                    comps_arr[i,j,0,k] = ccac.comps[0].flatten()[k - no_nans] # this is a^T*X
                    comps_arr[i,j,1,k] = ccac.comps[1].flatten()[k - no_nans] # this is b^T*Y
                    comps_arr[j,i,0,k] = comps_arr[i,j,0,k] # mirror results
                    comps_arr[j,i,1,k] = comps_arr[i,j,1,k] # mirror results


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
    ang_rel = xr.Dataset(data_vars = {'ang_rel': (['first_st', 'second_st'], ang_rel_arr)},
                         coords = {'first_st': stations,
                                   'second_st': stations})
    ang_abs = xr.Dataset(data_vars = {'ang_abs': (['first_st', 'second_st', 'index', 'ab'], ang_abs_arr)},
                         coords = {'first_st': stations,
                                   'second_st': stations,
                                   'index': range(num_cp),
                                   'ab': ['a', 'b']})
    # build Dataset from comps
    comps = xr.Dataset(data_vars = {'comps': (['first_st', 'second_st', 'uv', 'index'], comps_arr)},
                       coords = {'first_st': stations,
                                 'second_st': stations,
                                 'uv': ['u', 'v'],
                                 'index': range(num_cp)})

    # merge Datasets
    res = xr.merge([coeffs, weights, ang_rel, ang_abs, comps])

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

    # detrend input Dataset, remove NAs
    ds = sad.detrend(ds)
    ds = ds.dropna(dim = 'time')

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


def inter_phase_dir_corr(ds, station1, station2, wind_start1, wind_start2, win_len=128, components=None):
    """
    Calculates the CCA between two stations for two windows.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This is used to calculate the correlations.
    station1 and station2: float
        Stations you want to have a corellogram comparing, station1 remains fixed whilst the window is shifted for station2.
    wind_start1 and wind_start2: int
        The indexes of the windows you want to comapre.
    win_len: float, default 128
        The length of the window applied on the data.

    Returns
    -------
    cca_coeffs: float
        The first CCA coefficient.
    """
    # check if readings are provided
    if components is None:
        components = ['N', 'E', 'Z']

    num_comp = len(components)

    data = sad.window(ds,win_len)

    data1 = data.measurements.loc[dict(station = station1)][dict(win_start = wind_start1)]
    data2 = data.measurements.loc[dict(station = station2)][dict(win_start = wind_start2)]
    comb_st = xr.concat([data1, data2], dim = 'component')
    comb_st = comb_st.dropna(dim = 'win_rel_time', how = 'any')
    first_st = comb_st[:, 0:num_comp]
    second_st = comb_st[:, num_comp:2*num_comp]
    # run cca, suppress rcca output
    temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
    ccac = temp_cca.train([first_st, second_st])
    cca_coeffs = ccac.cancorrs[0]

    return cca_coeffs
