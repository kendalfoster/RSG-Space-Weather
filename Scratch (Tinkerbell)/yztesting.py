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



import spaceweather.supermag as sm

from spaceweather.visualisation import globes as svg
from spaceweather.visualisation import animations as sva

import numpy as np

station_components = sm.mag_csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",
                            MLT = True, MLAT = True)

t = station_components.time[1]
# list_of_stations = station_components.station


svg.plot_data_globe_colour(station_components, t, list_of_stations = None, ortho_trans = (0, 0))
# plots N and E components of the vector readings for a single time step t
# by default it plots data from all stations fed to it in station_readings unless
# specified otherwise in list_of_stations.
# ortho_trans specifies the angle from which we see the plot(earth) at.
# if left at default, yz.auto_ortho(list_of_stations) centres the view on the centre of all stations in list_of_stations.




sva.data_globe_gif(station_components, time_start = 0, time_end = 10, ortho_trans = (0, 0), colour = False, file_name = "sandra")
#makes sandra.gif in the /gif folder


sm.plot_connections_globe(station_components, adj_matrix = fake_data, ortho_trans = (0, 0), t = None, list_of_stations = None)
#plots connections between stations.
#for now it expects a 2d adjacency matrix as input but i will add code to make it do 3d(time on 3rd axis) as well
