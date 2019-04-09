## Packages
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt


## Function to restructure the SuperMAG data as a Dataset (xarray)
#       inputs: csv_file- SuperMAG data as csv file, downloaded from SuperMAG website
#               readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLAT- input False if the magnetic latitude column is NOT included, default is inclusion of the column
#               MLT- input False if the magnetic local time column is NOT included, default is inclusion of the column
#       output: Dataset with the SuperMAG data easily accessible,
#                       time is first dimension (ie, axis=0 for numpy commands)
#                       data is accessible via measurements
#

def mag_csv_to_Dataset(csv_file, readings=None, MLT=None, MLAT=None):
        data = pd.read_csv(csv_file)

        if readings is None:
                readings = ['N', 'E', 'Z']

        if ((MLAT is None) | (MLAT is True)):
                # sort stations by magnetic latitude (from north to south) and create MLAT array
                stations = data['IAGA'].unique()
                num_st = len(stations)
                mlat_arr = np.vstack((stations,np.zeros(num_st))).transpose()
                for i in range(0,num_st):
                        mlat_arr[i,1] = data['MLAT'].loc[data['IAGA'] == stations[i]].mean()
                mlat_arr = sorted(mlat_arr, key=lambda x: x[1], reverse=True)
                stations = [i[0] for i in mlat_arr]
                mlats = [round(i[1],4) for i in mlat_arr]
        elif MLAT is False:
                stations = data['IAGA'].unique()


        # build the DataArray
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




        # build Dataset from all the possible components
        if ((MLAT is None) | (MLAT is True)):
                ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading', 'station'], da),
                                             'mlats': (['station'], mlats)},
                                coords = {'time': times,
                                          'reading': readings,
                                          'station': stations})
        elif MLAT is False:
                ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading', 'station'], da)},
                                coords = {'time': times,
                                          'reading': readings,
                                          'station': stations})

        return ds


ds1 = mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv")
ds1












## Function to plot the readings like on the SuperMAG website
#       input: dataset output from mag_data_to_Dataset function
#       output: series of plots, one per station, of the readings

def plot_mag_data(ds):
        ds.measurements.plot.line(x='time', hue='reading', col='station', col_wrap=1)
