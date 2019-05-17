'''done:
plot stations
plot vector readings + gifs
plot connections (partial)
auto ortho_trans based on stations plotted
'''


'''todo:
plot connections - list_of_stations, t
colour code arrows to improve readability - map long, lat onto 2d grid of colours
colour code vertex connections similar to Dods

improve efficiency of gif fn if possible
make gifs
remove redundancies in plot_stations and plot_data_globe
incorporate MLAT, MLT as outlined by IGRF? make sure same version as kendal and all other data
'''



'''before running install cartopy using "conda install -c conda-forge cartopy" '''


from cartopy.feature.nightshade import Nightshade
# import spaceweather.analysis.supermag as sm
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
# import spaceweather.vi
from spaceweather.visualisation import globes as svg
import spaceweather.visualisation.animations as sva
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime

import pandas as pd

station_readings = sad.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

test = sac.cca(station_readings)

datetime.utcnow()
t = station_readings.time[1]
t.item()

test = pd.to_datetime(t)

dt64 = t.data
type(dt64)

dt64.item().astype(datetime)

svg.plot_data_globe_colour(station_readings, t.data, ortho_trans = (-90, 90))

sva.data_globe_gif(station_readings, time_end = 100)
