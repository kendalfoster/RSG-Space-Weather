## Packages

import numpy as np
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat
import spaceweather.visualisation.heatmaps as svh


ds = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv")



def supermag(csv_file=None, ds=None, thr_meth='Dods', win_len=128, **kwargs):

    return adj_mat
