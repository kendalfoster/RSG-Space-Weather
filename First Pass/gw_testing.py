import os
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
from scipy import signal
from numpy import hstack
import lib.generate_model_data as gen_data

pwd()
os.chdir('/home/gwatkins/mathsys_course/RSG_project/RSG-Space-Weather'
)
pwd()

scratch_ds = gen_data.generate_one_day_time_series('2001-04-03', '08:00:00', 30, 4, [0, 0.25, 0.5])

scratch_N = scratch_ds.measurements.loc[:,'N','XXX']
scratch_E = scratch_ds.measurements.loc[:,'E','XXX']
scratch_Z = scratch_ds.measurements.loc[:,'Z','XXX']

scratch_ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)


scratch_ds.measurements[480:510,:,:].plot.line(x='time', hue='component', col='station', col_wrap=1)
