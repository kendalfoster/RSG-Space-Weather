"""
========
Contents
========

Supermag
--------
supermag

Analysis
--------
- cca
- data_funcs
- gen_data
- threshold

Visualisation
-------------
- animations
- globes
- heatmaps
- lines
- spectral_analysis
"""


# need to be in RSG-Space-Weather folder
pwd()

## Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.gen_data as sag
import spaceweather.analysis.threshold as sat
import spaceweather.visualisation.animations as sva
import spaceweather.visualisation.static as svg
import spaceweather.visualisation.heatmaps as svh
import spaceweather.visualisation.lines as svl
import spaceweather.visualisation.spectral_analysis as svs
import spaceweather.supermag as sm
import xarray as xr
import numpy as np
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html





################################################################################
####################### Supermag ###############################################
################################################################################
ds1 = sad.csv_to_Dataset(csv_file="Data/20190403-00-22-supermag.csv", MLAT=True)
ds2 = ds1[dict(time = slice(177), station = range(4))]
test = sm.supermag(ds = ds2)
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
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv", MLAT = True)

##### cca ----------------------------------------------------------------------
X = ds1[dict(station = 0)].measurements.values
Y = ds1[dict(station = 1)].measurements.values
cca_ex = sac.cca(X, Y)

##### cca_angles ---------------------------------------------------------------
ds2 = ds1[dict(time = slice(75), station = slice(4))]
ccca_ang = sac.cca_angles(ds = ds2)

##### lag_mat_pair -------------------------------------------------------------
ds3 = ds1[dict(time = slice(177))] # slice must be at least win_len+2*lag_range
lag_mat = sac.lag_mat_pair(ds = ds3)
lag_mat2 = sac.lag_mat_pair(ds3, station1 = 'EKP', station2 = 'BLC',
                            lag_range = 10, win_len = 128, plot = False)

##### lag_mat ------------------------------------------------------------------
ds4 = ds1[dict(time = slice(177), station = range(4))]
lag_mat = sac.lag_mat(ds4)
lag_mat2 = sac.lag_mat(ds= ds4, lag_range = 8, win_len = 120)
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
ds1 = sad.csv_to_Dataset(csv_file="Data/20190403-00-22-supermag.csv", MLAT=True)
ds2 = ds1[dict(time = slice(177), station = range(4))]

##### threshold ----------------------------------------------------------------
lags = np.array([-2,0,1,3])

thresh = sat.threshold(ds = ds2, lags = lags)

##### max_corr_lag -------------------------------------------------------------
max_corr = sat.max_corr_lag(ds = ds2, lag_range = 10)

##### adj_mat ------------------------------------------------------------------
adj_mat = sat.adj_mat(ds = ds2, win_len = 128, lag_range = 10)

##### corr_lag_mat -------------------------------------------------------------
corr_lag_mat = sat.corr_lag_mat(ds = ds2)
################################################################################





################################################################################
####################### Visualisation ##########################################
################################################################################


####################### static #################################################
ds1 = sad.csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv", MLAT = True)
ds2 = ds1[dict(time = slice(150), station = range(4))]

##### csv_to_coords ------------------------------------------------------------
station_info = svg.csv_to_coords()

##### auto_ortho ---------------------------------------------------------------
list_of_stations = ds1.station
aut_orth = svg.auto_ortho(list_of_stations)

##### plot_stations ------------------------------------------------------------
list_of_stations = ds1.station
aut_orth = svg.auto_ortho(list_of_stations)
plot_of_stations = svg.plot_stations(list_of_stations = list_of_stations,
                                     ortho_trans = aut_orth)

##### plot_data_globe ----------------------------------------------------------
data_globe = svg.plot_data_globe(ds1)
data_globe2 = svg.plot_data_globe(ds = ds1, list_of_stations = ds1.station[2:6].values)
data_globe2 = svg.plot_data_globe(ds = ds1, list_of_components = ['N', 'Z'])
data_globe3 = svg.plot_data_globe(ds = ds1, t = 4)
data_globe4 = svg.plot_data_globe(ds = ds1, t = ds1.time[4].values)
data_globe5 = svg.plot_data_globe(ds = ds1, daynight=False)
data_globe6 = svg.plot_data_globe(ds1, colour=True)
data_globe7 = svg.plot_data_globe(ds1, color=True)

##### plot_connections_globe ---------------------------------------------------
import numpy as np
a_m = np.array([[np.nan,     1.,     1.,     1.],
                [np.nan, np.nan,     0.,     1.],
                [np.nan, np.nan, np.nan,     1.],
                [np.nan, np.nan, np.nan, np.nan]])
globe_conn = svg.plot_connections_globe(adj_matrix = a_m, ds = ds2)

##### plot_lag_network ---------------------------------------------------------
adj_mat = sat.adj_mat(ds2)
lag_net = svg.plot_lag_network(adj_mat[dict(win_start=3)])
################################################################################


####################### animations #############################################
ds1 = sad.csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv", MLAT = True)
ds2 = ds1[dict(time = slice(150), station = range(4))]

##### data_globe_gif -----------------------------------------------------------
sva.data_globe_gif(ds = ds2,
                   filepath = 'Scratch (Tinkerbell)/data_gif',
                   filename = 'globe_data',
                   colour = True)

##### connections_globe_gif ----------------------------------------------------
adj_mat = sat.adj_mat(ds = ds2, win_len = 128, lag_range = 10)
sva.connections_globe_gif(adj_mat_ds = adj_mat,
                          filepath = 'Scratch (Tinkerbell)/connections_gif',
                          filename = 'globe_connections')

##### lag_mat_gif_time ---------------------------------------------------------
lag_mat = sac.lag_mat(ds2)
sva.lag_mat_gif_time(lag_ds = lag_mat,
                     filepath = 'Scratch (Tinkerbell)/lag_mat_gif',
                     filename = 'lag_mat')

##### lag_network_gif ----------------------------------------------------------
adj_mat = sat.adj_mat(ds2)
sva.lag_network_gif(adj_matrix_ds = adj_mat,
                    filepath = 'Scratch (Tinkerbell)/lag_network_gif',
                    filename = 'lag_network')

##### corr_thresh_gif ----------------------------------------------------------
corr_lag_mat = sat.corr_lag_mat(ds2)
sva.corr_thresh_gif(corr_thresh_ds = corr_lag_mat,
                    filepath = 'Scratch (Tinkerbell)/corr_thresh_gif',
                    filename = 'corr_thresh')
################################################################################


####################### heatmaps ###############################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv", MLAT = True)

##### plot_adj_mat -------------------------------------------------------------
## currently not in use
fig = svh.plot_adj_mat(adj_mat = sat.adj_mat(ds = ds1[dict(time = slice(40))], thr_ds = ds1),
                       stations = ds1.station.values,
                       rns = range(len(ds1.station.values)))

##### plot_lag_mat_pair (correlogram) ------------------------------------------
ds2 = ds1[dict(time = slice(177))] # slice must be at least win_len+2*lag_range
lag_mat_pair = sac.lag_mat_pair(ds2, station1 = 'TAL', station2 = 'BLC')
fig = svh.plot_lag_mat_pair(lag_mat_pair = lag_mat_pair,
                            time_win = lag_mat_pair.time_win.values,
                            lag = lag_mat_pair.lag.values)

##### plot_lag_mat_time --------------------------------------------------------
ds2 = ds1[dict(time = slice(177))] # slice must be at least win_len+2*lag_range
lag_mat = sac.lag_mat(ds2)
lm = lag_mat[dict(time_win = 4, lag = 4, win_start = 4)]
lag_mat_fig = svh.plot_lag_mat_time(lm)

##### plot_corr_thresh ---------------------------------------------------------
ds2 = ds1[dict(time = slice(177), station = range(4))]
corr_lag_mat = sat.corr_lag_mat(ds2)
corr_thresh = svh.plot_corr_thresh(corr_lag_mat[dict(win_start=3)])
################################################################################


####################### lines ##################################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv", MLAT = True)

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
