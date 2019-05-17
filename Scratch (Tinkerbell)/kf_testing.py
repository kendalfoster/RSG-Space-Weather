## Packages
import xarray as xr
import numpy as np
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.gen_data as sag
import spaceweather.analysis.threshold as sat
import spaceweather.visualisation.animations as sva
import spaceweather.visualisation.globes as svg
import spaceweather.visualisation.heatmaps as svh
import spaceweather.visualisation.lines as svl
import spaceweather.visualisation.spectral_analysis as svs
import spaceweather.rcca as rcca




ccac_test = sac.cca_coeffs(ds)



#### This method works for the cca coefficients and components ####

ds = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")
components=['N', 'E', 'Z']

# check if more than one station in Dataset
stations = ds.station.values
num_st = len(stations)
if num_st <= 1:
    return 'Error: only one station in Dataset'

# other constants
num_ws = len(components)
num_cp = len(ds.time)
times = ds.time.values
ab = ['a', 'b']
uv = ['u', 'v']

# detrend input Dataset
ds = sad.detrend(ds)

for i in range(0, num_st-1): # first station, initialize the DataArray
    st_1 = ds.measurements.loc[dict(station = stations[i])]
    st_2 = ds.measurements.loc[dict(station = stations[i+1])]

    # remove NaNs from data (nans will mess up cca)
    both_st = xr.concat([st_1, st_2], dim = 'component')
    both_st = both_st.dropna(dim = 'time', how = 'any')
    temp_times = both_st.time.values
    st_1 = both_st[:, 0:num_ws]
    st_2 = both_st[:, num_ws:2*num_ws]

    # run cca, suppress rcca output
    temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
    ccac = temp_cca.train([st_1, st_2])

    ## store cca attributes ##
    # coefficients
    coeffs_arr = np.array([ccac.cancorrs[0].flatten()])
    coeffs_fs = xr.DataArray(data = coeffs_arr)
    coeffs_fs = coeffs_fs.assign_coords(first_st = i, second_st = i+1)
    coeffs_fs = coeffs_fs.squeeze(dim = ['dim_0', 'dim_1'], drop = True)

    # (cca) components
    comps_arr = np.array([ccac.comps[0].flatten(), ccac.comps[1].flatten()])
    comps_fs = xr.DataArray(data = comps_arr,
                            coords = [uv, temp_times],
                            dims = ['uv','time'])
    comps_fs = comps_fs.assign_coords(first_st = i, second_st = i+1)

    # loop through all but the last of the other stations ----------------------
    if num_st >= 3: # check if there are at least three stations
        for j in range(i+2, num_st): # second station in the pair
            st_2 = ds.measurements.loc[dict(station = stations[j])]

            # remove NaNs from data (nans will mess up cca)
            both_st = xr.concat([st_1, st_2], dim = 'component')
            both_st = both_st.dropna(dim = 'time', how = 'any')
            temp_times = both_st.time.values
            st_1 = both_st[:, 0:num_ws]
            st_2 = both_st[:, num_ws:2*num_ws]

            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([st_1, st_2])

            ## store cca attributes and merge into master DataArray ##
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


        if i==0:
            coeffs = coeffs_fs
            comps = comps_fs
        elif i < num_st-2:
            coeffs = xr.concat([coeffs, coeffs_fs], dim = 'first_st')
            comps = xr.concat([comps, comps_fs], dim = 'first_st')
        else: # i = num_st-1; ie the last loop
            # coefficients
            dummy = xr.DataArray(data = 0)
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            coeffs_dum = xr.concat([coeffs_fs, dummy], dim = 'second_st')
            coeffs = xr.concat([coeffs, coeffs_dum], dim = 'first_st')
            coeffs = coeffs.transpose('first_st', 'second_st')
            # (cca) components
            dum = np.zeros(shape = (2, len(temp_times)))
            dummy = xr.DataArray(data = dum,
                                 coords = [uv, temp_times],
                                 dims = ['uv','time'])
            dummy = dummy.assign_coords(first_st = num_st-2, second_st = num_st-2)
            comps_dum = xr.concat([comps_fs, dummy], dim = 'second_st')
            comps = xr.concat([comps, comps_dum], dim = 'first_st')
            comps = comps.transpose('first_st', 'second_st', 'uv', 'time')

### adjust the coordinates ###
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

### mirror results across the main diagonal ###
for i in range(0, num_st):
    for j in range(i+1, num_st):
        coeffs.values[j,i] = coeffs.values[i,j]
        comps.values[j,i,:,:] = comps.values[j,i,:,:]

#### label the coordinates ###
coeffs = coeffs.assign_coords(first_st = stations, second_st = stations)
comps = comps.assign_coords(first_st = stations, second_st = stations)
#-------------------------------------------------------------------------------
