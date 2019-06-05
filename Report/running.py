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

event_ds = sad.csv_to_Dataset('Report/CSV Files/event-1997-11-05.csv', MLAT=True)
event_am = sat.adj_mat(event_ds)
event_am.to_netcdf(path = 'Report/Saved Datasets/event-1997-11-05-adj-mat.nc')
