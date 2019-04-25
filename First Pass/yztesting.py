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



import lib.supermag as sm
import numpy as np

station_components = sm.mag_csv_to_Dataset(csv_file = "First Pass/poster_supermag_data.csv",
                            MLT = True, MLAT = True)

t = station_components.time[1]
list_of_stations = station_components.station


sm.plot_data_globe(station_components, t, list_of_stations = None, ortho_trans = (0, 0))
# plots N and E components of the vector readings for a single time step t
# by default it plots data from all stations fed to it in station_readings unless
# specified otherwise in list_of_stations.
# ortho_trans specifies the angle from which we see the plot(earth) at.
# if left at default, yz.auto_ortho(list_of_stations) centres the view on the centre of all stations in list_of_stations.




sm.data_globe_gif(station_components, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra")
#makes sandra.gif in the /gif folder


#generating fake adjacency matrix
N = 9
# length = 50
b = np.random.randint(-2000,2000,size=(N,N))


b_symm = (b + b.T)/2

fake_data = b_symm < 0



sm.plot_connections_globe(station_components, adj_matrix = fake_data, ortho_trans = (0, 0), t = None, list_of_stations = None)
#plots connections between stations.
#for now it expects a 2d adjacency matrix as input but i will add code to make it do 3d(time on 3rd axis) as well









## for the poster
ds2 = sm.mag_csv_to_Dataset(csv_file = "First Pass/poster_supermag_data.csv",
                            MLT = True, MLAT = True)
ds2w = ds2.loc[dict(time = slice('2001-03-05T12:00', '2001-03-05T14:00'))]
adj_mat = sm.mag_adj_mat(ds=ds2, ds_win=ds2w, n0=0.25)
sandy = sm.plot_connections_globe(ds2, adj_matrix = adj_mat.cca_coeffs, ortho_trans = (0, 0), t = None, list_of_stations = None)
sandy.savefig('First Pass/poster_globe.png', transparent=True)
