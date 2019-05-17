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



#### This method works for the cca coefficients ####

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

# detrend input Dataset
ds = sad.detrend(ds)

for i in range(0, num_st-1): # first station, initialize the DataArray
    st_1 = ds.measurements.loc[dict(station = stations[i])]
    st_2 = ds.measurements.loc[dict(station = stations[i+1])]

    # remove NaNs from data (nans will mess up cca)
    both_st = xr.concat([st_1, st_2], dim = 'component')
    both_st = both_st.dropna(dim = 'time', how = 'any')
    st_1 = both_st[:, 0:num_ws]
    st_2 = both_st[:, num_ws:2*num_ws]

    # run cca, suppress rcca output
    temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
    ccac = temp_cca.train([st_1, st_2])

    ## store cca attributes ##
    # coeffs
    coeffs_arr = np.array([ccac.cancorrs[0].flatten()])
    coeffs_fs = xr.DataArray(data = coeffs_arr)
    coeffs_fs = coeffs_fs.assign_coords(first_st = i)
    coeffs_fs = coeffs_fs.assign_coords(second_st = i+1)
    coeffs_fs = coeffs_fs.squeeze(dim = ['dim_0', 'dim_1'], drop = True)

    # loop through all but the last of the other stations ----------------------
    # check if there are at least three stations
    if num_st >= 3:
        for j in range(i+2, num_st): # second station
            st_2 = ds.measurements.loc[dict(station = stations[j])]

            # remove NaNs from data (nans will mess up cca)
            both_st = xr.concat([st_1, st_2], dim = 'component')
            both_st = both_st.dropna(dim = 'time', how = 'any')
            st_1 = both_st[:, 0:num_ws]
            st_2 = both_st[:, num_ws:2*num_ws]

            # run cca, suppress rcca output
            temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
            ccac = temp_cca.train([st_1, st_2])

            ## store cca attributes and merge into master DataArray ##
            # coeffs
            cfs_arr = np.array([ccac.cancorrs[0].flatten()])
            cfs = xr.DataArray(data = cfs_arr)
            cfs = cfs.assign_coords(first_st = i)
            cfs = cfs.assign_coords(second_st = j)
            cfs = cfs.squeeze(dim = ['dim_0', 'dim_1'], drop = True)

            coeffs_fs = xr.concat([coeffs_fs, cfs], dim = 'second_st')

        if i==0:
            coeffs = coeffs_fs
        elif i < num_st-2:
            coeffs = xr.concat([coeffs, coeffs_fs], dim = 'first_st')
        else: # i = num_st-1; ie the last loop
            dummy = xr.DataArray(data = 0)
            dummy = dummy.assign_coords(first_st = num_st-2)
            dummy = dummy.assign_coords(second_st = num_st-2)
            cdum = xr.concat([coeffs_fs, dummy], dim = 'second_st')
            coeffs = xr.concat([coeffs, cdum], dim = 'first_st')

# adjust the coordinates
blank = np.zeros(shape=(num_st, num_st))
blank_da = xr.DataArray(data = blank,
                        coords = [range(num_st), range(num_st)],
                        dims = ['first_st', 'second_st'])
coeffs = coeffs.combine_first(blank_da)

# mirror results across the main diagonal
for i in range(0, num_st):
    for j in range(i+1, num_st):
        coeffs.values[j,i] = coeffs.values[i,j]

# label the coordinates
coeffs = coeffs.assign_coords(first_st = stations)
coeffs = coeffs.assign_coords(second_st = stations)
#-------------------------------------------------------------------------------











##### Testing ------------------------------------------------------------------
dim1 = range(4)
dim2 = range(5)
dim1a = range(1,4)
dim2a = range(1,5)
dat0 = np.ones(shape=(4,5))
dat1 = np.zeros(shape=(4,5))
dat1a = np.zeros(shape=(3,4))

da0 = xr.DataArray(data = dat0,
                   coords = [dim1, dim2],
                   dims = ['dim1', 'dim2'])
da1 = xr.DataArray(data = dat1a,
                   coords = [dim1a, dim2a],
                   dims = ['dim1', 'dim2'])

da = xr.concat([da0, da1], dim = 'second_st')
