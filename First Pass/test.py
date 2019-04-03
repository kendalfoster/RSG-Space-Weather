## Packages
import numpy as np
import pandas as pd
import xarray as xr


## Import Data
data = pd.read_csv("20190403-00-22-supermag.csv")
readings = ['N', 'E', 'Z']


## Function to restructure the SuperMAG data as a Dataset (xarray)
#       inputs: data- SuperMAG data as a pandas dataframe using read_csv
#               meas- vector of characters representing measurements, default is ['N', 'E', 'Z']
def mag_data_to_Dataset(data, readings=None):
        if readings is None:
                readings = ['N', 'E', 'Z']

        stations = data['IAGA'].unique()
        num_st = stations.size
        times = data['Date_UTC'].unique()
        cols = np.append('Date_UTC', readings)

        # initialize DataArray (so we can append things to it later)
        temp_data = data[cols].loc[data['IAGA'] == stations[0]]
        temp_times = temp_data['Date_UTC'].unique()
        da = xr.DataArray(data = temp_data[readings],
                              coords = [temp_times, readings],
                              dims = ['time', 'reading']
                              )

        # loop through the stations and append each to master DataArray
        for i in stations[1:]:
                temp_data = data[cols].loc[data['IAGA'] == i]
                temp_times = temp_data['Date_UTC'].unique()
                temp_da = xr.DataArray(data = temp_data[readings],
                                       coords = [temp_times, readings],
                                       dims = ['time', 'reading']
                                       )
                da = xr.concat([da, temp_da], dim = 'station')
                da = da.transpose('time', 'reading', 'station')

        ds = xr.Dataset(data_vars = {'readings': (['time', 'reading', 'station'], da)},
                        coords = {'time': times,
                                  'reading': readings,
                                  'station': stations}
                        )

        return ds


## Testing
test_ds1 = mag_data_to_Dataset(data=data, readings=readings)
test_ds1

test_ds2 = mag_data_to_Dataset(data=data)
test_ds2
