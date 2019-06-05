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




quiet_day_am = xr.open_dataset('Report/Saved Datasets/quiet-day-adj-mat.nc')
sva.lag_network_gif(quiet_day_am, filepath='Report/Images/lag_network_gif/quiet_day', filename='quiet_day_lag_network')


event_am = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
sva.lag_network_gif(event_am, filepath='Report/Images/lag_network_gif/event', filename='event_lag_network')
