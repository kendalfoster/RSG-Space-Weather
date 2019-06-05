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


quiet_day_ds = sad.csv_to_Dataset('Report/CSV Files/quiet-day-1998-02-02.csv', MLAT=True)
svl.plot_mag_data(quiet_day_ds)
# quiet_day_am = sat.adj_mat(quiet_day_ds)
# quiet_day_am.to_netcdf(path = 'Report/Saved Datasets/quiet-day-adj-mat.nc')
quiet_day_am = xr.open_dataset('Report/Saved Datasets/quiet-day-adj-mat.nc')


event_ds = sad.csv_to_Dataset('Report/CSV Files/event-1997-11-05.csv', MLAT=True)
svl.plot_mag_data(event_ds)
# event_am = sat.adj_mat(event_ds)
# event_am.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
event_am = xr.open_dataset('Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
