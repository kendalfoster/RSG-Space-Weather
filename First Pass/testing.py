# need to be in RSG-Space-Weather folder
pwd()

## Packages
import lib.supermag as sm
import numpy as np


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
test_inter = sm.inter_st_cca(ds = ds1)
test_inter.cca_coeffs

## CCA between components in one station
test_intra = sm.intra_st_cca(ds = ds1, station = 'BSL')
test_intra

## CCA between components for all stations
test_all = sm.st_cca(ds = ds1)
test_all
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
