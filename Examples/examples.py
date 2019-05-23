# need to be in RSG-Space-Weather folder
pwd()

## Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.gen_data as sag
import spaceweather.analysis.threshold as sat
###import spaceweather.visualisation.animations as sva
###import spaceweather.visualisation.globes as svg
import spaceweather.visualisation.heatmaps as svh
import spaceweather.visualisation.lines as svl
import spaceweather.visualisation.spectral_analysis as svs
import spaceweather.supermag as sm
import numpy as np
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html





################################################################################
####################### Supermag ###############################################
################################################################################
ds1 = sad.csv_to_Dataset(csv_file="Data/20190403-00-22-supermag.csv", MLAT=True)
ds1 = ds1[dict(time = slice(147))]
test = sm.supermag(ds = ds1, MLAT = True)
test.adj_coeffs.values
################################################################################





################################################################################
####################### Analysis ###############################################
################################################################################


####################### data_funcs #############################################

##### csv_to_Dataset -----------------------------------------------------------
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                         MLT = True, MLAT = True)

ds2 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                         components = ['N', 'E', 'Z'],
                         MLT = True, MLAT = True)

# exclude MLT data
ds3 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                         MLT = False, MLAT = True)

# exclude MLAT data, order of stations should be different compared to above
ds4 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                         MLT = True, MLAT = False)

# exclude MLT and MLAT data, order of stations should also be different
ds5 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                         MLT = False, MLAT = False)

# CSV file only contains data for one station
ds_one = sad.csv_to_Dataset("Data/one-dik.csv", MLT=True, MLAT=True)


##### detrend ------------------------------------------------------------------
det = sad.detrend(ds=ds1)


##### window -------------------------------------------------------------------
ds1_win = sad.window(ds = ds1)
ds1_win_60 = sad.window(ds = ds1, win_len = 60)
ds1_slice_win = sad.window(ds1[dict(time=slice(0,10))], win_len=7)

ds2 = sad.csv_to_Dataset("Data/dik1996.csv", MLT = False, MLAT = False)
True in np.isnan(ds2.measurements)
ds2_win = sad.window(ds2)
################################################################################


####################### cca ####################################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")

##### cca ----------------------------------------------------------------------
cca_ex = sac.cca(ds = ds1)

##### cca_coeffs ---------------------------------------------------------------
coeffs_ex = sac.cca_coeffs(ds = ds1)
<<<<<<< HEAD
=======

## Correlogram
time, lag, corr, fig = svh.correlogram(ds1)
time, lag, corr, fig = svh.correlogram(ds1, station1 = 'EKP', station2 = 'DLR')
time, lag, corr, fig = svh.correlogram(ds1, lag_range = 3, win_len = 256)
>>>>>>> d26479a469606d3689e0a3cd59b66a68f6008a3d
################################################################################


####################### gen_data ###############################################
scratch_ds = sag.generate_one_day_time_series('2001-04-03', '08:00:00', 30, 4, [0, 0.25, 0.5],['XXX','YYY'])

scratch_N = scratch_ds.measurements.loc[:,'N','YYY']
scratch_E = scratch_ds.measurements.loc[:,'E','YYY']
scratch_Z = scratch_ds.measurements.loc[:,'Z','YYY']

scratch_ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)
scratch_ds.measurements[480:510,:,:].plot.line(x='time', hue='component', col='station', col_wrap=1)
################################################################################


####################### threshold ##############################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")

##### thresh_kf ----------------------------------------------------------------
thr_kf = sat.thresh_kf(ds = ds1)

##### thresh_dods --------------------------------------------------------------
thr_dods = sat.thresh_dods(ds = ds1)
thr_dods_25 = sat.thresh_dods(ds = ds1, n0 = 0.25)

##### threshold ----------------------------------------------------------------
thresh_kf = sat.threshold(ds = ds1, thr_meth = 'kf')
thresh_dods = sat.threshold(ds = ds1, thr_meth = 'Dods')
thresh_dods_25 = sat.threshold(ds = ds1, thr_meth = 'Dods', n0 = 0.25)

##### adj_mat ------------------------------------------------------------------
thr_ds = sad.csv_to_Dataset(csv_file = "Data/20010305-16-38-supermag.csv")
ds2 = thr_ds.loc[dict(time = slice('2001-03-05T12:00', '2001-03-05T14:00'))]
adj_mat = sat.adj_mat(ds=ds2, thr_ds=thr_ds, thr_meth='Dods', plot=False)
################################################################################





################################################################################
####################### Visualisation ##########################################
################################################################################


####################### animations ###############################################

################################################################################


####################### globes #################################################
station_components = sad.csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv")

t = station_components.time[1]
list_of_stations = station_components.station


svg.plot_data_globe(station_components, t, list_of_stations = None, ortho_trans = (0, 0))
# plots N and E components of the vector readings for a single time step t
# by default it plots data from all stations fed to it in station_readings unless
# specified otherwise in list_of_stations.
# ortho_trans specifies the angle from which we see the plot(earth) at.
# if left at default, yz.auto_ortho(list_of_stations) centres the view on the centre of all stations in list_of_stations.


sag.data_globe_gif(station_components, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra")
#makes sandra.gif in the /gif folder


#generating fake adjacency matrix
N = 9
# length = 50
b = np.random.randint(-2000,2000,size=(N,N))
b_symm = (b + b.T)/2
fake_data = b_symm < 0

svg.plot_connections_globe(station_components, adj_matrix = fake_data, ortho_trans = (0, 0), t = None, list_of_stations = None)
#plots connections between stations.
#for now it expects a 2d adjacency matrix as input but i will add code to make it do 3d(time on 3rd axis) as well
################################################################################


####################### heatmaps ###############################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")

##### plot_adj_mat -------------------------------------------------------------
fig = svh.plot_adj_mat(adj_mat = sat.adj_mat(ds = ds1[dict(time = slice(40))], thr_ds = ds1),
                       stations = ds1.station.values,
                       rns = range(len(ds1.station.values)))

##### correlogram --------------------------------------------------------------
ds2 = ds1[dict(time = slice(177))] # slice must be at least win_len+2*lag_range
time, lag, corr, fig = svh.correlogram(ds2)
time, lag, corr, fig = svh.correlogram(ds2, station1 = 'EKP', station2 = 'DLR')
time, lag, corr, fig = svh.correlogram(ds2, lag_range = 3, win_len = 64)
################################################################################


####################### lines ##################################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")

##### plot_mag_data ------------------------------------------------------------
svl.plot_mag_data(ds=ds1)

## extra code for editing titles of plots
import matplotlib.pyplot as plt
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


####################### spectral_analysis ######################################
gw_ds = sag.generate_one_day_time_series('2001-04-03', '08:00:00', 30, 4, [0, 0.25, 0.5],['XXX','YYY'])
gw_ps = gw_ds[dict(time=slice(480,510))].loc[dict(station='XXX', component='N')]
gw_ts = gw_ds.loc[dict(station = 'XXX', component = 'N')]

svs.power_spectrum(ts=gw_ps)
svs.power_spectrum(ds=gw_ds[dict(time=slice(480,510))], station='XXX', component='N')

svs.spectrogram(ts=gw_ts)
svs.spectrogram(ds=gw_ds, station='XXX', component='N')
################################################################################
