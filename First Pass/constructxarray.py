# need to be in RSG-Space-Weather folder
pwd()

## Packages
import numpy as np
import pandas as pd
import scipy.signal as scg
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import sys
import os
import lib.supermag as sm
import lib.rcca as rcca


################################################################################
########## Constructing xarray data structures ##########

#------------------ From Numpy Arrays ------------------------------------------
# define some vectors for later ease
times = [2011, 2012]
readings = ['N', 'E', 'Z']
stations = ['TAL', 'BLC', 'EKS', 'BLS']

# Numpy arrray
george = np.ones(shape = (2,3,4))

# Numpy array -> xarray DataArray
george_da = xr.DataArray(data = george,
                         coords = [times, readings, stations],
                         dims = ['time', 'reading', 'station'])
george_da

# Numpy array -> xarray Dataset
george_ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading', 'station'], george)},
                       coords = {'time': times,
                                 'reading': readings,
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


#------------------ Concatenating xarray Data Structures -----------------------
# concatenation allows for "stacking" arrays of different lengths, filling with nans

# define some vectors for later ease
times = [2011, 2012, 2013, 2014]
readings = ['N', 'E', 'Z']
stations = ['TAL', 'BLC', 'EKS', 'BLS', 'EKP']

# Numpy Arrays
jorge1 = np.ones(shape = (4,3))
jorge = np.zeros(shape = (3,2))

### Concatenate "Stack" DataArrays
# Numpy array -> xarray DataArray
jorge_da = xr.DataArray(data = jorge1,
                        coords = [times, readings],
                        dims = ['time', 'reading'])
for i in stations[1:]:
    temp = xr.DataArray(data = jorge,
                        coords = [times[1:], readings[:2]],
                        dims = ['time', 'reading'])
    jorge_da = xr.concat([jorge_da, temp], dim = 'station')

jorge_da = jorge_da.assign_coords(station = stations)
jorge_da2ds = jorge_da.to_dataset(name = 'measurements')
jorge_da2ds

### Concatenate "Stack" Datasets
# Numpy array -> xarray Dataset
jorge_ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading'], jorge1)},
                      coords = {'time': times,
                                'reading': readings})
for i in stations[1:]:
    temp = xr.Dataset(data_vars = {'measurements': (['time', 'reading'], jorge)},
                          coords = {'time': times[1:],
                                    'reading': readings[:2]})
    jorge_ds = xr.concat([jorge_ds, temp], dim = 'station')
jorge_ds = jorge_ds.assign_coords(station = stations)
jorge_ds

### Rearrange coordinates (supermag.py functions assume 'time' first)
jorge_da2ds = jorge_da2ds.transpose('time', 'reading', 'station')
jorge_ds = jorge_ds.transpose('time', 'reading', 'station')
################################################################################
