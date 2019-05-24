import spaceweather.visualisation.static as svs
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat
import numpy as np
import pandas as pd




ds1 = sad.csv_to_Dataset(csv_file = "Old Presentations/Poster/poster_supermag_data.csv", MLAT = True)
ds2 = ds1[dict(time = slice(150), station = range(4))]
# adj_mat = sat.adj_mat(ds = ds2, win_len = 128, lag_range = 10)
a_m = np.array([[np.nan,     1.,     1.,     1.],
                [np.nan, np.nan,     0.,     1.],
                [np.nan, np.nan, np.nan,     1.],
                [np.nan, np.nan, np.nan, np.nan]])

globe_conn = svs.plot_connections_globe(ds = ds2, adj_matrix = a_m)
