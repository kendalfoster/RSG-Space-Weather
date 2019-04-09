## Packages
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt


################################################################################
### Function to restructure the SuperMAG data as a Dataset (xarray)
#       inputs: csv_file- SuperMAG data as csv file, downloaded from SuperMAG website
#               readings- vector of characters representing measurements, default is ['N', 'E', 'Z']
#               MLAT- input True if the magnetic latitude column is included, default is non-inclusion of the column
#               MLT- input True if the magnetic local time column is included, default is non-inclusion of the column
#       output: Dataset with the SuperMAG data easily accessible,
#                       time is first dimension (ie, axis=0 for numpy commands)
#                       data is accessible in array format via output.measurements
#

def mag_csv_to_Dataset(csv_file, readings=None, MLT=None, MLAT=None):
        # get universally needed things
        data = pd.read_csv(csv_file)
        times = pd.to_datetime(data['Date_UTC'].unique())

        #-----------------------------------------------------------------------
        #---------- optional arguments -----------------------------------------
        #-----------------------------------------------------------------------

        # check if readings are provided
        if readings is None:
                readings = ['N', 'E', 'Z']

        # if MLAT is included, sort and make Dataset
        if MLAT is True:
                # sort stations by magnetic latitude (from north to south)
                stations = data['IAGA'].unique()
                num_st = len(stations)
                mlat_arr = np.vstack((stations,np.zeros(num_st))).transpose()
                for i in range(0,num_st):
                        mlat_arr[i,1] = data['MLAT'].loc[data['IAGA'] == stations[i]].mean()
                mlat_arr = sorted(mlat_arr, key=lambda x: x[1], reverse=True)
                stations = [i[0] for i in mlat_arr]
                mlats = [round(i[1],4) for i in mlat_arr]
                # build MLAT Dataset, for merging later
                ds_mlat = xr.Dataset(data_vars = {'mlats': (['station'], mlats)},
                                     coords = {'time': times,
                                               'reading': readings,
                                               'station': stations})
        elif MLAT is not True:
                stations = data['IAGA'].unique()

        # if MLT (Magnetic Local Time) is included, make a Dataset
        if MLT is True:
                # initialize DataArray (so we can append things to it later)
                cols_mlt = ['Date_UTC', 'MLT']
                temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == stations[0]]
                temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
                mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                                  coords = [temp_times_mlt],
                                  dims = ['time'])
                # loop through the stations and append each to master DataArray
                for i in stations[1:]:
                        temp_data_mlt = data[cols_mlt].loc[data['IAGA'] == i]
                        temp_times_mlt = pd.to_datetime(temp_data_mlt['Date_UTC'].unique())
                        temp_mlt = xr.DataArray(data = temp_data_mlt['MLT'],
                                          coords = [temp_times_mlt],
                                          dims = ['time'])
                        mlt = xr.concat([mlt, temp_mlt], dim = 'station')
                        mlt = mlt.transpose('time', 'station')
                # build MLT Dataset, for merging later
                ds_mlt = xr.Dataset(data_vars = {'mlts': (['time', 'station'], mlt)},
                                    coords = {'time': times,
                                              'reading': readings,
                                              'station': stations})



        #-----------------------------------------------------------------------
        #---------- build the main DataArray of the measurements ---------------
        #-----------------------------------------------------------------------

        # initialize DataArray (so we can append things to it later)
        cols = np.append('Date_UTC', readings)
        temp_data = data[cols].loc[data['IAGA'] == stations[0]]
        temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
        da = xr.DataArray(data = temp_data[readings],
                          coords = [temp_times, readings],
                          dims = ['time', 'reading'])

        # loop through the stations and append each to master DataArray
        for i in stations[1:]:
                temp_data = data[cols].loc[data['IAGA'] == i]
                temp_times = pd.to_datetime(temp_data['Date_UTC'].unique())
                temp_da = xr.DataArray(data = temp_data[readings],
                                       coords = [temp_times, readings],
                                       dims = ['time', 'reading'])
                da = xr.concat([da, temp_da], dim = 'station')
                da = da.transpose('time', 'reading', 'station')

        # build Dataset from readings
        ds = xr.Dataset(data_vars = {'measurements': (['time', 'reading', 'station'], da)},
                        coords = {'time': times,
                                  'reading': readings,
                                  'station': stations})



        #-----------------------------------------------------------------------
        #---------- build the final DataArray from optional arguments ----------
        #-----------------------------------------------------------------------

        # include MLT
        if MLT is True:
                ds = xr.merge([ds, ds_mlt])

        # include MLAT
        if MLAT is True:
                ds = xr.merge([ds, ds_mlat])


        return ds
################################################################################





################################################################################
### Function to plot the readings like on the SuperMAG website
#       input: dataset output from mag_data_to_Dataset function
#       output: series of plots, one per station, of the readings

def plot_mag_data(ds):
        ds.measurements.plot.line(x='time', hue='reading', col='station', col_wrap=1)
################################################################################
