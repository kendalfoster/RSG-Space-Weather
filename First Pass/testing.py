# need to be in RSG-Space-Weather folder
pwd()

## Packages
import lib.supermag as sm
import numpy as np
import lib.rcca as rcca
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html



################################################################################
####################### Restructuring the SuperMAG Data ########################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
ds1

ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            components = ['N', 'E', 'Z'],
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
################################################################################


sm.corellogram(ds2, "BLC", "TAL", 5, 256)

sm.corellogram_max(d)

################################################################################
####################### Plotting ###############################################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
sm.plot_mag_data(ds=ds1)


## extra code for editing titles of plots
ds1 = ds1.loc[dict(station = slice('BLC'))]
stations = ds1.station.loc[dict(station = slice('BLC'))].values
components = ds1.component.values
## all of below code must be run simultaneously
g = ds1.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1, add_legend=False)
for i, ax in enumerate(g.axes.flat):
   ax.set_title(stations[i], fontsize=30)

plt.legend(labels=components, loc='right', title='Component', title_fontsize='x-large', fontsize=20)
plt.draw()
################################################################################




################################################################################
####################### Detrending #############################################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

det = sm.mag_detrend(ds=ds1)
det
################################################################################




################################################################################
####################### Windowing ##############################################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

ds1_win = sm.window(ds = ds1)
ds1_win
ds1_win_60 = sm.window(ds = ds1, win_len = 60)
ds1_win_60
ds1_win_slice = sm.window(ds = ds1[dict(time=slice(0,10))], win_len = 3)
ds1_win_slice.measurements.loc[dict(station = 'TAL')]
ds1_win_slice[dict(window = 0)]
################################################################################




################################################################################
####################### Canonical Correlation Analysis #########################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

## CCA between stations
test_inter = sm.cca_coeffs(ds = ds1)
test_inter
################################################################################




################################################################################
####################### Thresholding ###########################################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

#--- Canonical Correlation ---
## KF Threshold
thresh_kf = sm.mag_thresh_kf(ds = ds1)
thresh_kf.thresholds.values

## Dods-style Threshold
thresh_dods = sm.mag_thresh_dods(ds = ds1, n0 = 0.25)
thresh_dods.thresholds.values

## Adjacency Matrix Functions
ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20010305-16-38-supermag.csv",
                            MLT = True, MLAT = True)
ds2w = ds2.loc[dict(time = slice('2001-03-05T12:00', '2001-03-05T14:00'))]

adj_mat = sm.mag_adj_mat(ds=ds2, ds_win=ds2w, n0=0.25)
adj_mat2 = sm.print_mag_adj_mat(ds=ds2, ds_win=ds2w, n0=0.25)

#--- Phase Correlation ---
## KF Threshold

## Dods-style Threshold

################################################################################




################################################################################
####################### Constructing the Network ###############################
ds1 = sm.mag_csv_to_Dataset(csv_file = "First Pass/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)
ds1 = ds1[dict(time=slice(0,148))]

con_ds1 = sm.construct_network(ds = ds1, win_len = 128, n0 = 0.25)
con_ds2 = sm.construct_network(ds = ds1)
################################################################################




################################################################################
####################### Visualizing the Network ################################
station_components = sm.mag_csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv",
                            MLT = True, MLAT = True)

t = station_components.time[1]
list_of_stations = station_components.station


sm.plot_data_globe(station_components, t, list_of_stations = None, ortho_trans = (0, 0))
# plots N and E components of the vector readings for a single time step t
# by default it plots data from all stations fed to it in station_readings unless
# specified otherwise in list_of_stations.
# ortho_trans specifies the angle from which we see the plot(earth) at.
# if left at default, yz.auto_ortho(list_of_stations) centres the view on the centre of all stations in list_of_stations.




sm.data_globe_gif(station_components, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra")
#makes sandra.gif in the /gif folder


#generating fake adjacency matrix
N = 9
# length = 50
b = np.random.randint(-2000,2000,size=(N,N))


b_symm = (b + b.T)/2

fake_data = b_symm < 0



sm.plot_connections_globe(station_components, adj_matrix = fake_data, ortho_trans = (0, 0), t = None, list_of_stations = None)
#plots connections between stations.
#for now it expects a 2d adjacency matrix as input but i will add code to make it do 3d(time on 3rd axis) as well

################################################################################




################################################################################
####################### Generating Model Data ##################################
scratch_ds = sm.generate_one_day_time_series('2001-04-03', '08:00:00', 30, 4, [0, 0.25, 0.5],['XXX','YYY'])

scratch_N = scratch_ds.measurements.loc[:,'N','YYY']
scratch_E = scratch_ds.measurements.loc[:,'E','YYY']
scratch_Z = scratch_ds.measurements.loc[:,'Z','YYY']

scratch_ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)


scratch_ds.measurements[480:510,:,:].plot.line(x='time', hue='component', col='station', col_wrap=1)

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
