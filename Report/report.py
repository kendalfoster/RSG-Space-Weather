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


##### Read Data ----------------------------------------------------------------
quiet_day_ds = sad.csv_to_Dataset('Report/CSV Files/quiet-day-1998-02-02.csv', MLAT=True)
svl.plot_mag_data(quiet_day_ds)

event_ds = sad.csv_to_Dataset('Report/CSV Files/event-1997-11-05.csv', MLAT=True)
svl.plot_mag_data(event_ds)


##### Generate Adjacency Matrices ----------------------------------------------
# quiet_day_am = sat.adj_mat(quiet_day_ds)
# quiet_day_am.to_netcdf(path = 'Report/Saved Datasets/quiet-day-adj-mat.nc')
quiet_day_am = xr.open_dataset('Report/Saved Datasets/quiet-day-adj-mat.nc')

# event_am = sat.adj_mat(event_ds)
# event_am.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
event_am = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-adj-mat.nc')


##### Generate Lag Matrices for One Station Pair -------------------------------
quiet_day_lm = sac.lag_mat_pair(quiet_day_ds)
quiet_day_lm.to_netcdf(path = 'Report/Saved Datasets/quiet-day-lag-mat-pair.nc')
quiet_day_lm = xr.open_dataset('Report/Saved Datasets/quiet-day-lag-mat-pair.nc')

event_lm = sac.lag_mat_pair(event_ds)
event_lm.to_netcdf(path = 'Report/Saved Datasets/event-lag-mat-pair.nc')
event_lm = xr.open_dataset('Report/Saved Datasets/event-lag-mat-pair.nc')


##### Network Parameters -------------------------------------------------------
# quiet_day_net_params = san.network_params(quiet_day_am, avg=True, norm=True)
# quiet_day_net_params.to_netcdf(path = 'Report/Saved Datasets/quiet-day-net-params.nc')
quiet_day_net_params = xr.open_dataset('Report/Saved Datasets/quiet-day-net-params.nc')

event_net_params = san.network_params(event_am, avg=True, norm=True)
event_net_params.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-net-params.nc')
event_net_params = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-net-params.nc')


##### Plot Lag Networks --------------------------------------------------------
quiet_day_lag_net = svg.plot_lag_network(quiet_day_am[dict(win_start=0)])
sva.lag_network_gif(quiet_day_am, filepath='Report/Images/lag_network_gif/quiet_day', filename='quiet_day_lag_network')

event_lag_net = svg.plot_lag_network(event_am[dict(win_start=0)])
sva.lag_network_gif(event_am, filepath='Report/Images/lag_network_gif/event', filename='event_lag_network')


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
