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

from PIL import Image

cca_ang_ds = xr.open_dataset('Report/Saved Datasets/quiet-day-cca-ang.nc')
sva.cca_ang_gif(quiet_day_cca_ang, a_b = 'a',
                filepath = 'Report/Images/cca_ang_gif',
                filename = 'cca_ang')

filepath = 'Report/Images/cca_ang_gif'
filename = 'cca_ang'
im_filepath = filepath + '/images_for_giffing'
# get constants
times = cca_ang_ds.time.values
num_times = len(times)

# initialize the list of names of image files
names = []

# plot the connections for each win_start value in the adjacency matrix
for i in range(num_times):
    im_name = im_filepath + '/%s.png' %i
    names.append(im_name) # add name of image file to list
len(names)

images = []
for n in names:
    images.append(Image.open(n))

# make gif file and save it in filepath
images[0].save(filepath + '/%s.gif' %filename,
               save_all = True,
               append_images = images[1:],
               duration = 50, loop = 0)
