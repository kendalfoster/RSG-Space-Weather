'''
This file is meant for letting things run in the background.
'''
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


quiet_day_ds = sad.csv_to_Dataset('Report/CSV Files/quiet-day-1998-02-02.csv', MLAT=True)
event_ds = sad.csv_to_Dataset('Report/CSV Files/event-1997-11-05.csv', MLAT=True)



quiet_day_corr_lag = sat.corr_lag_mat(quiet_day_ds.loc[dict(time=slice('1998-02-02T11:00','1998-02-02T17:00'))])
quiet_day_corr_lag.to_netcdf(path = 'Report/Saved Datasets/quiet-day-corr-lag-part2.nc')
quiet_day_corr_lag = xr.open_dataset('Report/Saved Datasets/quiet-day-corr-lag-part2.nc')

event_corr_lag = sat.corr_lag_mat(event_ds.loc[dict(time=slice('1997-11-05T9:00','1997-11-05T15:00'))])
event_corr_lag.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-corr-lag-part2.nc')
event_corr_lag = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-corr-lag-part2.nc')



values = quiet_day_corr_lag.corr_thresh.values
values[values > 0] = 1
values[values <= 0] = 0
quiet_day_corr_lag.values = values
qd_cr_am = quiet_day_corr_lag.rename(corr_thresh = 'adj_coeffs')
qd_cc = san.cluster_coeff(qd_cr_am)
np.nanmean(qd_cc.time_avg_local_cc)

values = event_corr_lag.corr_thresh.values
values[values > 0] = 1
values[values <= 0] = 0
event_corr_lag.values = values
e_cr_am = event_corr_lag.rename(corr_thresh = 'adj_coeffs')
e_cc = san.cluster_coeff(e_cr_am)
np.nanmean(e_cc.time_avg_local_cc)
