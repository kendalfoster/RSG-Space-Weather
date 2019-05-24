# need to be in RSG-Space-Weather folder
pwd()

## Packages
import numpy as np
import xarray as xr # if gives error, just rerun


################################################################################
####################### Constructing xarray data structures ####################

#---------------------- From Numpy Arrays --------------------------------------
# define some vectors for later ease
times = [2011, 2012]
components = ['N', 'E', 'Z']
stations = ['TAL', 'BLC', 'EKS', 'BLS']

# Numpy arrray
george = np.ones(shape = (2,3,4))

# Numpy array -> xarray DataArray
george_da = xr.DataArray(data = george,
                         coords = [times, components, stations],
                         dims = ['time', 'component', 'station'])
george_da

# Numpy array -> xarray Dataset
george_ds = xr.Dataset(data_vars = {'measurements': (['time', 'component', 'station'], george)},
                       coords = {'time': times,
                                 'component': components,
                                 'station': stations})
george_ds
# each coord requires
    # 1) a name that matches a dimension name from data_vars
    # 2) a vector whose length matches the 'length' of the data along the specified dimension

# xarray DataArray -> xarray Dataset
george_da2ds = george_da.to_dataset(name = 'measurements')
george_da2ds

# a Dataset is basically a wrapper for one or more DataArrays that share coordinates
    # hence, each data_var in a Dataset has a (unique) name
    # in our SuperMAG Dataset, there are 3 data_vars representing 3 underlying DataArrays
    # so to access one, we use Dataset.name_of_underlying_DataArray
george_ds.measurements
george_da2ds.measurements


#---------------------- Concatenating xarray Data Structures -------------------
# concatenation allows for "stacking" arrays of different lengths, filling with nans

# define some vectors for later ease
times = [2011, 2012, 2013, 2014]
components = ['N', 'E', 'Z']
stations = ['TAL', 'BLC', 'EKS', 'BLS', 'EKP']

# Numpy Arrays
jorge1 = np.ones(shape = (4,3))
jorge = np.zeros(shape = (3,2))

### Concatenate "Stack" DataArrays
# Numpy array -> xarray DataArray
jorge_da = xr.DataArray(data = jorge1,
                        coords = [times, components],
                        dims = ['time', 'component'])
for i in stations[1:]:
    temp = xr.DataArray(data = jorge,
                        coords = [times[1:], components[:2]],
                        dims = ['time', 'component'])
    jorge_da = xr.concat([jorge_da, temp], dim = 'station')

jorge_da = jorge_da.assign_coords(station = stations)
jorge_da2ds = jorge_da.to_dataset(name = 'measurements')
jorge_da2ds

### Concatenate "Stack" Datasets
# Numpy array -> xarray Dataset
jorge_ds = xr.Dataset(data_vars = {'measurements': (['time', 'component'], jorge1)},
                      coords = {'time': times,
                                'component': components})
for i in stations[1:]:
    temp = xr.Dataset(data_vars = {'measurements': (['time', 'component'], jorge)},
                          coords = {'time': times[1:],
                                    'component': components[:2]})
    jorge_ds = xr.concat([jorge_ds, temp], dim = 'station')
jorge_ds = jorge_ds.assign_coords(station = stations)
jorge_ds

### Rearrange coordinates (supermag.py functions assume 'time' first)
jorge_da2ds = jorge_da2ds.transpose('time', 'component', 'station')
jorge_ds = jorge_ds.transpose('time', 'component', 'station')
################################################################################




################################################################################
####################### Saving and Loading xarray Data Structures ##############
### Make a Dataset
times = [2011, 2012]
components = ['N', 'E', 'Z']
stations = ['TAL', 'BLC', 'EKS', 'BLS']
arr = np.ones(shape = (2,3,4))

example_da = xr.DataArray(data = arr,
                          coords = [times, components, stations],
                          dims = ['time', 'component', 'station'])

example_ds = xr.Dataset(data_vars = {'measurements': (['time', 'component', 'station'], arr)},
                        coords = {'time': times,
                                  'component': components,
                                  'station': stations})

# xarray Data Structures are stored as NetCDF files
#---------------------- Saving xarray Data Structures --------------------------
example_ds.to_netcdf(path = 'Examples/example_ds.nc')
example_da.to_netcdf(path = 'Examples/example_da.nc')

#---------------------- Loading xarray Data Structures -------------------------
another_ds = xr.open_dataset('Examples/example_ds.nc')
another_da = xr.open_dataarray('Examples/example_da.nc')
################################################################################
