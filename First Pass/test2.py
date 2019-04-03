## Preamble
import numpy as np
import pandas as pd
import xarray as xr


## Import Data
data = pd.read_csv("20190403-00-22-supermag.csv")
data = data.loc[0:35]
readings = ['N', 'E', 'Z']
stations = data['IAGA'].unique()
num_st = stations.size
times = data['Date_UTC'].unique()

da = xr.DataArray(data = data[readings].loc[data['IAGA'] == stations[0]],
                      coords = [times, readings],
                      dims = ['time', 'reading']
                      )

for i in stations[1:]:
        print(i)
        
        temp_da = xr.DataArray(data = data[readings].loc[data['IAGA'] == i],
                              coords = [times, readings],
                              dims = ['time', 'reading']
                              )
        da = xr.concat([da, temp_da], dim = 'station')
        da = da.transpose('time', 'reading', 'station')

da

stations

temp_da = xr.DataArray(data = data[readings].loc[data['IAGA'] == 'BSL'], dims = ['time', 'reading'])
temp_da['time'].size

for i in range(1,temp_da['time'].size):
        if (temp_da['time'][i] - temp_da['time'][i-1]) > 9:
                print(temp_da['time'][i])


dump = np.zeros(9)
for i in range(0,stations.size):
        dump[i] = data['N'].loc[data['IAGA'] == stations[i]].size

dump

data[readings].isnull().any().any()

aaa = data[readings].loc[data['IAGA'] == 'BSL']
aaa
