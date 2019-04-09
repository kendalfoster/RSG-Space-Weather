# need to be in RSG-Space-Weather folder

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
data = pd.read_csv("First Pass/20190403-00-22-supermag.csv")

## Restructure SuperMAG Data
ds = sm.mag_data_to_Dataset(data=data)

blc_n = ds.readings.loc[dict(station = 'BLC')].loc[dict(reading = 'N')]
blc_e = ds.readings.loc[dict(station = 'BLC')].loc[dict(reading = 'E')]
blc_z = ds.readings.loc[dict(station = 'BLC')].loc[dict(reading = 'Z')]

blc_n1 = np.reshape(blc_n.values, newshape=[len(blc_n.values),1])
blc_e1 = np.reshape(blc_e.values, newshape=[len(blc_e.values),1])
blc_z1 = np.reshape(blc_z.values, newshape=[len(blc_z.values),1])

blc = ds.readings.loc[dict(station = 'BLC')]
tal = ds.readings.loc[dict(station = 'TAL')]
ran = ds.readings.loc[dict(station = 'RAN')]


cca_ds = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
cca_ds.train([blc_n1, blc_e1, blc_z1])
cca_ds.cancorrs


cca_ds.train([blc_z1, blc_e1])


# now turn the above into a function that operates on the whole dataset, ds





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
