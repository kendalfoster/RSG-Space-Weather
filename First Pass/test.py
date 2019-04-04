## Packages
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt


## Function to restructure the SuperMAG data as a Dataset (xarray)
#       inputs: data- SuperMAG data as a pandas dataframe imported using read_csv
#               readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLAT- input False if the magnetic latitude column is NOT included, default is inclusion of the column
#       output: Dataset with the SuperMAG data easily accessible

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

                # initialize DataArray (so we can append things to it later)
                times = pd.to_datetime(data['Date_UTC'].unique())
                cols = np.append('Date_UTC', readings)

                temp_data = data[cols].loc[data['IAGA'] == stations[0]]
                temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
                da = xr.DataArray(data = temp_data[readings],
                                      coords = [temp_times, readings],
                                      dims = ['time', 'reading']
                                      )

                # loop through the stations and append each to master DataArray
                for i in stations[1:]:
                        temp_data = data[cols].loc[data['IAGA'] == i]
                        temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
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

                # initialize DataArray (so we can append things to it later)
                times = pd.to_datetime(data['Date_UTC'].unique())
                cols = np.append('Date_UTC', readings)

                temp_data = data[cols].loc[data['IAGA'] == stations[0]]
                temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
                da = xr.DataArray(data = temp_data[readings],
                                      coords = [temp_times, readings],
                                      dims = ['time', 'reading']
                                      )

                # loop through the stations and append each to master DataArray
                for i in stations[1:]:
                        temp_data = data[cols].loc[data['IAGA'] == i]
                        temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
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



## Function to plot the readings like on the SuperMAG website
#       input: dataset output from mag_data_to_Dataset function
#       output: series of plots, one per station, of the readings

def plot_mag_data(ds):
        ds.readings.plot.line(x='time', col='station', col_wrap=1)




## Testing
data = pd.read_csv("20190403-00-22-supermag.csv")
readings = ['N', 'E', 'Z']


ds1 = mag_data_to_Dataset(data=data, readings=readings)
ds1

ds2 = mag_data_to_Dataset(data=data)
ds2

ds3 = mag_data_to_Dataset(data=data, MLAT=False)
ds3 # should be different order of stations to all the others

ds4 = mag_data_to_Dataset(data=data, MLAT=True)
ds4


plot_mag_data(ds=ds2)
