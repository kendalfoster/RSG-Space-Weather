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


quiet_day_cca_ang = xr.open_dataset('Report/Saved Datasets/quiet-day-cca-ang.nc')
sva.cca_ang_gif(quiet_day_cca_ang, a_b = 'a',
                filepath = 'Report/Images/cca_ang_gif',
                filename = 'cca_ang')
