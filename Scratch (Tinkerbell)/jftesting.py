### jftesting.py


# need to be in RSG-Space-Weather folder
pwd()

## Packages
import lib.supermag as sm
import numpy as np
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html



################################################################################
####################### Restructuring the SuperMAG Data ########################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            components = ['N', 'E', 'Z'],
                            MLT = True, MLAT = True)

# exclude MLT data
ds3 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = False, MLAT = True)

# exclude MLAT data, order of stations should be different compared to above
ds4 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = False)

# exclude MLT and MLAT data, order of stations should also be different
ds5 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = False, MLAT = False)

# CSV file only contains data for one station
ds_one = sm.mag_csv_to_Dataset("First Pass/one-dik.csv", MLT=True, MLAT=True)
################################################################################





import seaborn as sns

ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            components = ['N', 'E', 'Z'],
                            MLT = True, MLAT = True)

det = sm.mag_detrend(ds=ds2)
x, y, z = sm.corellogram(det, "TAL", "RAN" , lag_range=3, win_len=200)



import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr

plot = plt.pcolormesh(x,y,z, norm = colors.LogNorm(vmin = 0.999, vmax = 1))


sm.inter_phase_dir_corr(det, "RAN", "TAL",258, 3,128)


ds = det
station1 = "RAN"
station2 = "TAL"
wind_start1 = 0
wind_start2 = 0
win_len = 256
components = ['N', 'E', 'Z']

num_comp = len(components)

data = sm.window(ds,win_len)
data

data1 = data.measurements.loc[dict(station = station1)][dict(win_start = wind_start1)]
data2 = data.measurements.loc[dict(station = station2)][dict(win_start = wind_start2)]
comb_st = xr.concat([data1, data2], dim = 'component')
comb_st = comb_st.dropna(dim = 'win_rel_time', how = 'any')
first_st = comb_st[:, 0:num_comp]
second_st = comb_st[:, num_comp:2*num_comp]
# run cca, suppress rcca output
temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
ccac = temp_cca.train([first_st, second_st])
cca_coeffs = ccac.cancorrs[0]
ccac
ccac.cancorrs[0]
temp_cca


a = sm.cca_coeffs(det)
a.cca_coeffs

ds2
