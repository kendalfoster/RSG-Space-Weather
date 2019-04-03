## Preamble
import numpy as np
import pandas as pd
import xarray as xr


## Import Data
data = pd.read_csv("20190403-00-22-supermag.csv")
readings = ['N', 'E', 'Z']
stations = data['IAGA'].unique()
num_st = stations.size
times = data['Date_UTC'].unique()

times.size
data['Date_UTC'].size


da = xr.DataArray(data = data[readings].iloc[0:num_st],
                      coords = [stations, readings],
                      dims = ['station', 'reading']
                      )

for i in range(1,times.size):
        print(i)
        temp_da = xr.DataArray(data = data[readings].iloc[i*num_st:(i+1)*num_st],
                              coords = [stations, readings],
                              dims = ['station', 'reading']
                              )
        da = xr.concat([da, temp_da], dim = 'time')
        da = da.transpose('station', 'reading', 'time')

ds = xr.Dataset(
                data_vars = {'readings': (['station', 'reading', 'time'], da)},
                coords = {'station': stations,
                          'reading': readings,
                          'time': times}
                )
ds




data1 = xr.DataArray(data = data[['N', 'E', 'Z']].iloc[0:9],
                      coords = [stations, readings],
                      dims = ['station', 'reading']
                      )
data1

data2 = xr.DataArray(data = data[['N', 'E', 'Z']].iloc[9:18],
                      coords = [stations, readings],
                      dims = ['station', 'reading']
                      )
data2

ready_data = xr.concat([data1, temp_da], dim='time')
ready_data = ready_data.transpose('station', 'reading', 'time')
ready_data

times = data['Date_UTC'].unique()
times[0:2]


ds = xr.Dataset(
                data_vars = {'readings': (['station', 'reading', 'time'], ready_data)},
                coords = {'station': stations,
                          'reading': readings,
                          'time': times[0:2]}
                )
ds




temp1 = 15 + 8 * np.random.randn(9, 3)
temp = xr.DataArray(temp1)
temp = xr.concat([temp], dim='time')
temp = temp.transpose('dim_0', 'dim_1', 'time')
precip1 = 10 * np.random.rand(9, 3)
precip = xr.DataArray(precip1)
precip = xr.concat([precip], dim='time')
precip = precip.transpose('dim_0', 'dim_1', 'time')

lon = [[-99.83, -99.32, -99.83, -99.32], [-99.83, -99.32, -99.79, -99.23], [-99.83, -99.32, -99.79, -99.23]]
lat = [[42.25, 42.21, 42.25, 42.21], [42.25, 42.21, 42.63, 42.59], [42.25, 42.21, 42.63, 42.59]]

tempds = xr.Dataset(
                     data_vars={'temperature': (['x', 'y', 'time'],  temp),
                                'precipitation': (['x', 'y', 'time'], precip)}
                     )
tempds
