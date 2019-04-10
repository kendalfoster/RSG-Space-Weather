# need to be in RSG-Space-Weather folder
pwd()

###############################################################################
########## supermag.py ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm


## Restructure SuperMAG Data
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
########## rcca.py ##########
import numpy as np
import pandas as pd
import xarray as xr # if gives error, just rerun
import matplotlib.pyplot as plt
import lib.supermag as sm
import lib.rcca as rcca


## Import Data
ds = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

blc_n = ds.measurements.loc[dict(station = 'BLC')].loc[dict(reading = 'N')]
blc_e = ds.measurements.loc[dict(station = 'BLC')].loc[dict(reading = 'E')]
blc_z = ds.measurements.loc[dict(station = 'BLC')].loc[dict(reading = 'Z')]

blc_n1 = np.reshape(blc_n.values, newshape=[len(blc_n.values),1])
blc_e1 = np.reshape(blc_e.values, newshape=[len(blc_e.values),1])
blc_z1 = np.reshape(blc_z.values, newshape=[len(blc_z.values),1])

blc = ds.measurements.loc[dict(station = 'BLC')]
tal = ds.measurements.loc[dict(station = 'TAL')]
ran = ds.measurements.loc[dict(station = 'RAN')]
bsl = ds.measurements.loc[dict(station = 'BSL')]

cca_ds = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
cca_ds.train([blc, tal]).cancorrs[0]


xr.concat([blc, bsl], dim = 'cca_st')

aaa = xr.concat([blc,tal], dim = 'cca_st')
ccc = rcca.CCA(kernelcca = False, reg = 0., numCC = 1).train(aaa).cancorrs[0]


bbb = ds.measurements.loc[dict(station = stations[7])].values
rm_nan = np.isfinite(bbb)
rm_nan
rm_nan[:,0]
bbb[np.isfinite(bbb)].shape
bbb[~rm_nan]

x = [0,1,2,3]
1 in x
True in rm_nan

# now turn the above into a function that operates on the whole dataset, ds

def inter_st_cca(ds):
    ds = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                                MLT = True, MLAT = True)
    # universally necessary things
    stations = ds.station
    num_st = len(stations)

    # setup (triangular) array for the correlation coefficients
    cca_coeffs = np.zeros(shape = (num_st, num_st), dtype = float)

    # shrinking nested for loops to get all the pairs of stations
    for i in range(0, num_st-1):
        first_st = ds.measurements.loc[dict(station = stations[0])]
        False in np.isfinite(first_st)
        # test station for NaNs in the data (will mess up cca)
        first_st
        for j in range(i+1, num_st):
            second_st = ds.measurements.loc[dict(station = stations[1])]
            False in np.isfinite(second_st)
            comb_st = xr.concat([first_st, second_st], dim='reading')
            comb_st
            False in np.isfinite(comb_st)
            # test stations for NaNs in the data (will mess up cca)
            # sew the two DataArrays together along time dimension, and drop all rows with nan values
                # if Nans exist, remove the whole row of data from both stations and run cca

                # else run cca on original data
                cca_coeffs[i,j] =
                temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
                temp_cancorrs = temp_cca.train(comb_st).cancorrs
                temp_cancorrs



ds.measurements.loc[dict(station = 'BLC')]





##### from GitHub example
# Initialize number of samples
nSamples = 1000

# Define two latent variables (number of samples x 1)
latvar1 = np.random.randn(nSamples,)
latvar2 = np.random.randn(nSamples,)

# Define independent components for each dataset (number of observations x dataset dimensions)
indep1 = np.random.randn(nSamples, 4)
indep2 = np.random.randn(nSamples, 5)

# Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
data1 = 0.25*indep1 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2)).T
data2 = 0.25*indep2 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

# Split each dataset into two halves: training set and test set
train1 = data1[:nSamples//2]
train2 = data2[:nSamples//2]
test1 = data1[nSamples//2:]
test2 = data2[nSamples//2:]

train1.shape
train2.shape

# Create a cca object as an instantiation of the CCA object class.
nComponents = 4
cca = rcca.CCA(kernelcca = False, reg = 0., numCC = nComponents)

# Use the train() method to find a CCA mapping between the two training sets.
cca.train([train1, train2, train1])
cca.cancorrs.shape
cca.cancorrs

# Use the validate() method to test how well the CCA mapping generalizes to the test data.
# For each dimension in the test data, correlations between predicted and actual data are computed.
testcorrs = cca.validate([test1, test2])
testcorrs
#####################
###############################################################################
