# need to be in RSG-Space-Weather folder
pwd()

## Packages
import lib.supermag as sm
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt




################################################################################
########## Poster Thresholding Picture ##########
ds2 = sm.mag_csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv",
                            MLT = True, MLAT = True)
ds2w = ds2.loc[dict(time = slice('2001-03-05T15:25', '2001-03-05T17:25'))]

stations = ds2.station.values
num_st = len(stations)

cca = sm.inter_st_cca(ds=ds2w)
cca = cca.assign_coords(first_st = range(num_st))
cca = cca.assign_coords(second_st = range(num_st))

thresh = sm.mag_thresh_dods(ds=ds2)
thresh = thresh.assign_coords(first_st = range(num_st))
thresh = thresh.assign_coords(second_st = range(num_st))

adj_mat = cca - thresh.thresholds
adj_mat = adj_mat.assign_coords(first_st = range(num_st))
adj_mat = adj_mat.assign_coords(second_st = range(num_st))

# define new colormap
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmap = ListedColormap(newcolors, name='OrangeBlue')

# must run all following code simultaneously
fig = plt.figure(figsize=(10,8))
g = adj_mat.cca_coeffs.plot.pcolormesh(yincrease=False,
                                       cmap=newcmap,
                                       cbar_kwargs={'label': 'Correlation Coefficient - Threshold'})
plt.title('Adjacency Matrix', fontsize=30)
plt.xlabel('Station 1', fontsize=20)
plt.xticks(ticks=range(9), labels=stations, rotation=0)
plt.ylabel('Station 2', fontsize=20)
plt.yticks(ticks=range(9), labels=stations, rotation=0)
g.figure.axes[-1].yaxis.label.set_size(20)
plt.savefig('Old Presentations/Poster/adj_mat.png')
plt.show()
################################################################################




################################################################################
########## Poster Globe Picture ##########
ds2 = sm.mag_csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv",
                            MLT = True, MLAT = True)
ds2a = ds2.loc[dict(time = slice('2001-03-05T12:00', '2001-03-05T14:00'))]
adj_mat = sm.mag_adj_mat(ds=ds2, ds_win=ds2a, n0=0.25)
sandy = sm.plot_connections_globe(ds2, adj_matrix = adj_mat.cca_coeffs, ortho_trans = (0, 0), t = None, list_of_stations = None)
sandy.savefig('Old Presentations/Poster/poster_globe.png', transparent=True)
################################################################################
