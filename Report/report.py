## Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.gen_data as sag
import spaceweather.analysis.threshold as sat
import spaceweather.analysis.network as san
import spaceweather.visualisation.animations as sva
import spaceweather.visualisation.static as svg
import spaceweather.visualisation.heatmaps as svh
import spaceweather.visualisation.lines as svl
import spaceweather.visualisation.spectral_analysis as svs
import spaceweather.supermag as sm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


##### Read Data ----------------------------------------------------------------
quiet_day_ds = sad.csv_to_Dataset('Report/CSV Files/quiet-day-1998-02-02.csv', MLAT=True)
quiet_day_plot = quiet_day_ds.loc[dict(station='RAN')]
avg_meas = np.nanmean(np.nanmean(quiet_day_plot.measurements.values, axis=0))

event_ds = sad.csv_to_Dataset('Report/CSV Files/event-1997-11-05.csv', MLAT=True)
event_plot = event_ds.loc[dict(station='RAN')]

fig, axs = plt.subplots(2, figsize=(16,8))
# fig.suptitle('Big Title')
axs[0].set_title('Station RAN: Quiet Day 1998-02-02')
axs[1].set_title('Station RAN: Event Day 1997-11-05')
axs[0].set_ylabel('Measurement, nT')
axs[1].set_ylabel('Measurement, nT')
axs[1].set_xlabel('Time')
# handles = np.full(shape=3, fill_value=np.nan)
for i in range(3):
    axs[0].plot(quiet_day_plot.time.values, quiet_day_plot[dict(component=i)].measurements.values, label=i)
    axs[1].plot(event_plot.time.values, event_plot[dict(component=i)].measurements.values)
axs[0].legend(labels=['N', 'E', 'Z'], title='Component', bbox_to_anchor=(1.1, 0.1))
fig.savefig('Report/Images/quiet_vs_event_plot.png')
fig

qd_plot = svl.plot_mag_data(quiet_day_ds.loc[dict(station=['BLC', 'RAN'])])
e_plot = svl.plot_mag_data(event_ds.loc[dict(station=['BLC', 'RAN'])])


##### Generate Adjacency Matrices ----------------------------------------------
# quiet_day_am = sat.adj_mat(quiet_day_ds)
# quiet_day_am.to_netcdf(path = 'Report/Saved Datasets/quiet-day-adj-mat.nc')
quiet_day_am = xr.open_dataset('Report/Saved Datasets/quiet-day-adj-mat.nc')

# event_am = sat.adj_mat(event_ds)
# event_am.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
event_am = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-adj-mat.nc')


##### Generate Lag Matrices for One Station Pair -------------------------------
# quiet_day_lm = sac.lag_mat_pair(quiet_day_ds)
# quiet_day_lm.to_netcdf(path = 'Report/Saved Datasets/quiet-day-lag-mat-pair.nc')
quiet_day_lm = xr.open_dataset('Report/Saved Datasets/quiet-day-lag-mat-pair.nc')

# event_lm = sac.lag_mat_pair(event_ds)
# event_lm.to_netcdf(path = 'Report/Saved Datasets/event-lag-mat-pair.nc')
event_lm = xr.open_dataset('Report/Saved Datasets/event-lag-mat-pair.nc')


##### Plot Correlograms for One Station Pair -----------------------------------
qd_lmp_ds = quiet_day_ds.loc[dict(time=slice('1998-02-02T11:00','1998-02-02T17:00'), station = ['BLC', 'RAN'])]
quiet_day_lmp, quiet_day_correlogram = sac.lag_mat_pair(quiet_day_ds.loc[dict(station=['BLC','RAN'])], plot=True)
quiet_day_correlogram.savefig('Report/Images/quiet_day_correlogram.png')

ed_lmp_ds = event_ds.loc[dict(time=slice('1997-11-05T9:00','1997-11-05T15:00'), station = ['BLC', 'RAN'])]
event_lmp, event_correlogram = sac.lag_mat_pair(event_ds.loc[dict(station=['BLC','RAN'])], plot=True)
event_correlogram.savefig('Report/Images/event_correlogram.png')


##### Generate Correlation-Threshold Datasets ----------------------------------
quiet_day_corr_lag = sat.corr_lag_mat(quiet_day_ds.loc[dict(time=slice('1998-02-02T12:00','1998-02-02T14:28'))])
quiet_day_corr_lag.to_netcdf(path = 'Report/Saved Datasets/quiet-day-corr-lag-part.nc')
quiet_day_corr_lag = xr.open_dataset('Report/Saved Datasets/quiet-day-corr-lag-part.nc')

event_corr_lag = sat.corr_lag_mat(event_ds.loc[dict(time=slice('1997-11-05T11:30','1997-11-05T13:58'))])
event_corr_lag.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-corr-lag-part.nc')
event_corr_lag = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-corr-lag-part.nc')

qd_cr_plot = quiet_day_corr_lag.loc[dict(win_start = '1998-02-02T12:00')]
qd_corr_thresh_plot = svh.plot_corr_thresh(qd_cr_plot)
qd_corr_thresh_plot.savefig('Report/Images/quiet_day_corr_thresh.png')

e_cr_plot = event_corr_lag.loc[dict(win_start = '1997-11-05T11:30')]
e_corr_thresh_plot = svh.plot_corr_thresh(e_cr_plot)
e_corr_thresh_plot.savefig('Report/Images/event_cor_thresh.png')


##### Plot Adjacency Matrices --------------------------------------------------
values = qd_cr_plot.corr_thresh.values
values[values > 0] = 1
values[values <= 0] = 0
qd_cr_plot.values = values
qd_cr_am = qd_cr_plot.rename(corr_thresh = 'adj_coeffs')
qd_am_plot = svh.plot_adj_mat(qd_cr_am)
qd_am_plot = svh.plot_adj_mat(quiet_day_am.loc[dict(win_start = '1998-02-02T12:00')])
qd_am_plot.savefig('Report/Images/quiet_day_adj_mat.png')

values = e_cr_plot.corr_thresh.values
values[values > 0] = 1
values[values <= 0] = 0
e_cr_plot.values = values
e_cr_am = e_cr_plot.rename(corr_thresh = 'adj_coeffs')
event_am_plot = svh.plot_adj_mat(e_cr_am)
event_am_plot = svh.plot_adj_mat(event_am.loc[dict(win_start = '1997-11-05T11:30')])
event_am_plot.savefig('Report/Images/event_adj_mat.png')


##### Network Parameters -------------------------------------------------------
quiet_day_corr_lag = xr.open_dataset('Report/Saved Datasets/quiet-day-corr-lag-part2.nc')
values = quiet_day_corr_lag.corr_thresh.values
values[values > 0] = 1
values[values <= 0] = 0
quiet_day_corr_lag.values = values
qd_cr_am = quiet_day_corr_lag.rename(corr_thresh = 'adj_coeffs')
quiet_day_net_params = san.network_params(qd_cr_am, avg=True, norm=True)
quiet_day_net_params.data_vars
quiet_day_net_params.time_avg_local_cc

# quiet_day_net_params = san.network_params(quiet_day_am, avg=True, norm=True)
# quiet_day_net_params.to_netcdf(path = 'Report/Saved Datasets/quiet-day-net-params.nc')
quiet_day_net_params = xr.open_dataset('Report/Saved Datasets/quiet-day-net-params.nc')
qdnp = quiet_day_net_params.loc[dict(win_start=slice('1998-02-02T11:00','1998-02-02T17:00'), station = ['BLC', 'RAN'])]


event_corr_lag = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-corr-lag-part2.nc')
values = event_corr_lag.corr_thresh.values
values[values > 0] = 1
values[values <= 0] = 0
event_corr_lag.values = values
e_cr_am = event_corr_lag.rename(corr_thresh = 'adj_coeffs')
event_net_params = san.network_params(e_cr_am, avg=True, norm=True)
# event_net_params.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-net-params.nc')
# event_net_params = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-net-params.nc')
enp = event_net_params.loc[dict(win_start=slice('1997-11-05T9:00','1997-11-05T15:00'), station = ['BLC', 'RAN'])]
event_net_params.data_vars
event_net_params.time_avg_local_cc


##### Plot Lag Networks --------------------------------------------------------
quiet_day_lag_net = svg.plot_lag_network(quiet_day_am[dict(win_start=0)])
sva.lag_network_gif(quiet_day_am, filepath='Report/Images/lag_network_gif/quiet_day', filename='quiet_day_lag_network')
qd_ln = svg.plot_lag_network(qd_cr_am)
qd_ln.savefig('Report/Images/quiet_day_lag_network.png')

event_lag_net = svg.plot_lag_network(event_am[dict(win_start=0)])
sva.lag_network_gif(event_am, filepath='Report/Images/lag_network_gif/event', filename='event_lag_network')
e_ln = svg.plot_lag_network(e_cr_am)
e_ln.savefig('Report/Images/event_lag_network.png')


##### CCA Angles ---------------------------------------------------------------
# quiet_day_cca_ang = sac.cca_angles(quiet_day_ds)
# quiet_day_cca_ang.to_netcdf(path = 'Report/Saved Datasets/quiet-day-cca-ang.nc')
quiet_day_cca_ang = xr.open_dataset('Report/Saved Datasets/quiet-day-cca-ang.nc')
quiet_day_cca_ang_png = svl.plot_cca_ang_pair(quiet_day_cca_ang[dict(first_st=0, second_st=1)])
quiet_day_cca_ang_png.savefig('Report/Images/quiet_day_cca_ang.png')

# event_cca_ang = sac.cca_angles(event_ds)
# event_cca_ang.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-cca-ang.nc')
event_cca_ang = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-cca-ang.nc')
event_cca_ang_png = svl.plot_cca_ang_pair(event_cca_ang[dict(first_st=0, second_st=1)])
event_cca_ang_png.savefig('Report/Images/event_cca_ang.png')
