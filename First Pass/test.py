## Packages
import numpy as np
import pandas as pd
import xarray as xr


## Function to restructure the SuperMAG data as a Dataset (xarray)
#       inputs: data- SuperMAG data as a pandas dataframe imported using read_csv
#               readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLAT- input False if the magnetic latitude column is NOT included, default is inclusion of the column

def mag_data_to_Dataset(data, readings=None, MLAT=None):
        if readings is None:
                readings = ['N', 'E', 'Z']

        if ((MLAT is None) | (MLAT is True)):
                # sort stations by magnetic latitude (from north to south)
                stations = data['IAGA'].unique()
                num_st = stations.size
                stations = np.vstack((stations,np.zeros(num_st))).transpose()
                for i in range(0,num_st):
                        stations[i,1] = data['MLAT'].loc[data['IAGA'] == stations[i,0]].mean()

                stations = sorted(stations, key=lambda x: x[1], reverse=True)
                stations = [i[0] for i in stations]

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

        if MLAT is False:
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
data = pd.read_csv("20190403-00-22-supermag.csv")
readings = ['N', 'E', 'Z']


test_ds1 = mag_data_to_Dataset(data=data, readings=readings)
test_ds1

test_ds2 = mag_data_to_Dataset(data=data)
test_ds2

test_ds3 = mag_data_to_Dataset(data=data, MLAT=False)
test_ds3

test_ds4 = mag_data_to_Dataset(data=data, MLAT=True)
test_ds4
