'''
This file is meant for letting things run in the background.
'''
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


##### Data Vectors Gif ---------------------------------------------------------
quiet_day_ds = sad.csv_to_Dataset('Report/CSV Files/quiet-day-1998-02-02.csv', MLAT=True)
sva.data_globe_gif(quiet_day_ds, filepath='Report/Images/data_vectors_gif/quiet_day', filename='quiet_day_data_vectors')

event_ds = sad.csv_to_Dataset('Report/CSV Files/event-1997-11-05.csv', MLAT=True)
sva.data_globe_gif(event_ds, filepath='Report/Images/data_vectors_gif/event', filename='event_data_vectors')


##### Correlation-Threshold Gif ------------------------------------------------
quiet_day_corr_lag = xr.open_dataset('Report/Saved Datasets/quiet-day-corr-lag.nc')
sva.corr_thresh_gif(quiet_day_corr_lag, filepath='Report/Images/corr_thresh_gif/quiet_day', filename='quiet_day_corr_thresh')

event_corr_lag = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-corr-lag.nc')
sva.corr_thresh_gif(event_corr_lag, filepath='Report/Images/corr_thresh_gif/event', filename='event_corr_thresh')


##### Lag Network Gif ----------------------------------------------------------
quiet_day_am = xr.open_dataset('Report/Saved Datasets/quiet-day-adj-mat.nc')
sva.lag_network_gif(quiet_day_am, filepath='Report/Images/lag_network_gif/quiet_day', filename='quiet_day_lag_network')

event_am = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
sva.lag_network_gif(event_am, filepath='Report/Images/lag_network_gif/event', filename='event_lag_network')


##### CCA Angles Gif -----------------------------------------------------------
quiet_day_cca_ang = xr.open_dataset('Report/Saved Datasets/quiet-day-cca-ang.nc')
sva.cca_ang_gif(quiet_day_cca_ang, a_b='a', filepath='Report/Images/cca_ang_gif/quiet_day', filename='quiet_day_cca_ang')

event_cca_ang = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-cca-ang.nc')
sva.cca_ang_gif(event_cca_ang, a_b='a', filepath='Report/Images/cca_ang_gif/event', filename='event_cc_ang')
