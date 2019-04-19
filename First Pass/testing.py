# need to be in RSG-Space-Weather folder
pwd()

###############################################################################
########## Restructuring and Plotting the SuperMAG Data ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm


## Restructure SuperMAG
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
ds1

ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            readings = ['N', 'E', 'Z'],
                            MLT = True, MLAT = True)
ds2

ds3 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = False, MLAT = True)
ds3 # exclude MLT data

ds4 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = False)
ds4 # exclude MLAT data, order of stations should be different compared to above

ds4 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = False, MLAT = False)
ds4 # exclude MLT and MLAT data, order of stations should also be different

## Plot SuperMAG Data
sm.plot_mag_data(ds=ds1)
###############################################################################




###############################################################################
########## Canonical Correlation Analysis ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm
import lib.rcca as rcca

## Import and Restructure SuperMAG Data
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

## CCA between stations
test_inter = sm.inter_st_cca(ds = ds1);
test_inter.cca_coeffs

## CCA between readings in one station
test_intra = sm.intra_st_cca(ds = ds1, station = 'BSL')
test_intra

## CCA between readings for all stations
test_all = sm.st_cca(ds = ds1)
test_all
###############################################################################


read1 = ds1.measurements.loc[dict(station = "BSL")]
read2 = ds1.measurements.loc[dict(station = "BLC")]
read1.shape
read2.shape

merge = xr.concat([read1, read2], dim = 'station')
merge.shape

read1new = read1.dropna(dim = 'time', how = 'any')
read2new = read2.dropna(dim = 'time', how = 'any')
mergenew = merge.dropna(dim = 'time', how = 'any')
mergenew.shape

mergenew.measurements.loc[dict(station = "BSL")]

mergenew[0]
read1new

sm.cleans_na(ds1, "BSL", "BLC", readings=None)

def cleans_na(ds, station1, station2, readings=None):
    # check if readings are provided
    if readings is None:
        readings = ['N', 'E', 'Z']

    #Read data and merge
    read1 = ds1.measurements.loc[dict(station = station1)]
    read2 = ds1.measurements.loc[dict(station = station2)]
    merge = merge = xr.concat([read1, read2], dim = 'station')

    #Drop n/a values
    mergenew = merge.dropna(dim = 'time', how = 'any')

    #Split apart again
    data1 = mergenew[0]
    data2 = mergenew[1]

    return (data1, data2)

def window(ds, win_len=128):
    # create a rolling object
    ds_roll = ds.rolling(time=win_len).construct(window_dim='win_rel_time').dropna('time')
    # fix window coordinates
    ds_roll = ds_roll.assign_coords(win_rel_time = range(win_len))
    ds_roll = ds_roll.rename(dict(time = 'win_start'))
    # ensure the coordinates are in the proper order
    ds_roll = ds_roll.transpose('win_start', 'reading', 'station', 'win_rel_time')

    return ds_roll

window(ds1,128)


def inter_direction_cca(ds, station1, station2, readings=None):
     # check if readings are provided
      if readings is None:
          readings = ['N', 'E', 'Z']

      # universally necessary things
      num_read = len(readings)

      # get readings for the station
      data1, data2 = cleans_na(ds, station1, station2)

      # setup row array for the correlation coefficients
      cca_coeffs = np.zeros(shape = (1, num_read), dtype = float)

      #Calculate the cannonical correlation between the directional meaurements on each station
      for i in range(num_read):
          first_read = data1[:,i].data
          first_read = np.reshape(first_read, newshape=[len(first_read),1])

          second_read = data2[:,i].data
          second_read = np.reshape(second_read, newshape=[len(second_read),1])

          temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
          cca_coeffs[0,i] = abs(temp_cca.train([first_read, second_read]).cancorrs[0])
      return cca_coeffs

inter_direction_cca(ds1,"BSL","BLC")
