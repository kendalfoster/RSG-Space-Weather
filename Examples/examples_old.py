# need to be in RSG-Space-Weather folder
pwd()

## Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.gen_data as sag
import spaceweather.analysis.threshold as sat
import spaceweather.visualisation.animations as sva
import spaceweather.visualisation.globes as svg
import spaceweather.visualisation.heatmaps as svh
import spaceweather.visualisation.lines as svl
import spaceweather.visualisation.spectral_analysis as svs
import spaceweather.supermag as sm
import spaceweather.old as old
import numpy as np
# may need to install OpenSSL for cartopy to function properly
# I needed it on Windows, even though OpenSSL was already installed
# https://slproweb.com/products/Win32OpenSSL.html



################################################################################
####################### Analysis ###############################################
################################################################################


####################### threshold ##############################################
ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")

##### thresh_kf ----------------------------------------------------------------
thr_kf = old.thresh_kf(ds = ds1)

##### thresh_dods --------------------------------------------------------------
thr_dods = old.thresh_dods(ds = ds1)
thr_dods_25 = old.thresh_dods(ds = ds1, n0 = 0.25)

##### threshold ----------------------------------------------------------------
thresh_kf = old.threshold(ds = ds1, thr_meth = 'kf')
thresh_dods = old.threshold(ds = ds1, thr_meth = 'Dods')
thresh_dods_25 = old.threshold(ds = ds1, thr_meth = 'Dods', n0 = 0.25)

##### adj_mat ------------------------------------------------------------------
thr_ds = sad.csv_to_Dataset(csv_file = "Data/20010305-16-38-supermag.csv")
ds2 = thr_ds.loc[dict(time = slice('2001-03-05T12:00', '2001-03-05T14:00'))]
adj_mat = old.adj_mat(ds=ds2, thr_ds=thr_ds, thr_meth='Dods', plot=False)
################################################################################
