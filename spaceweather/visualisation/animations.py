"""
Contents
--------

- data_globe_gif
- anim_connections_globe
"""


## Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
## Local Packages
import spaceweather.visualisation.static as svs


def data_globe_gif(ds, filepath='data_gif', filename='globe_data',
                   list_of_stations=None, list_of_components=['N', 'E'],
                   ortho_trans=None, daynight=True, colour=False, **kwargs):
    '''
    Animates the data vectors on a globe over time with an optional shadow for
    nighttime and optional data colouration.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
        3-dimensional Dataset whose coordinates are first_st, second_st, time.
    filepath : str, optional
        File path for storing the image files and gif. Default is
        'data_gif' folder to be made in the current working directory.
    filename : str, optional
        File name for the gif, without file extension. Default is 'globe_data'.
    list_of_stations : list, optional
        List of stations in ds to be used on the plot.
    list_of_components : list, optional
        List of components in ds to be used on the plot. Must be of length 2.
    ortho_trans : tuple, optional
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.
    daynight : bool, optional
        Whether or not to include a shadow for nighttime. Default is True.
    colour : bool, optional
        Whether or not to colour the data vectors. Also accepts 'color' for
        Americans who can't spell properly.

    Returns
    -------
    .png
        png image files used to make the gif animation, saved in filepath/images_for_giffing.
    .gif
        gif animation of the png image files, saved in filepath/gif.
    '''

    # check filepaths
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    im_filepath = filepath + '/images_for_giffing'
    if not os.path.exists(im_filepath):
        os.makedirs(im_filepath)

    # check filename
    if '.' in filename:
        if len(filename) > 4:
            filename = filename[:-4] # remove file extension
        else:
            print('Error: please input filename without file extension')
            return 'Error: please input filename without file extension'

    # get contstants
    if list_of_stations is None:
        list_of_stations = ds.station.values
    list_of_times = ds.time.values
    num_times = len(list_of_times)

    # check ortho_trans
    if ortho_trans is None:
        ortho_trans = svs.auto_ortho(list_of_stations)

    # initialize the list of names of image files
    names = []

    # plot the data vectors for each time in the Dataset
    for i in range(num_times):
        fig = svs.plot_data_globe(ds = ds,
                                  list_of_stations = list_of_stations,
                                  list_of_components = list_of_components,
                                  t = i,
                                  ortho_trans = ortho_trans,
                                  daynight = daynight,
                                  colour = colour,
                                  **kwargs)
        im_name = im_filepath + '/%s.png' %i
        fig.savefig(im_name) # save image file
        names.append(im_name) # add name of image file to list

    # append plots to each other
    images = []
    for n in names:
        images.append(Image.open(n))

    # make gif file and save it in filepath
    images[0].save(filepath + '/%s.gif' %filename,
                   save_all = True,
                   append_images = images[1:],
                   duration = 50, loop = 0)


def connections_globe_gif(adj_mat_ds,
                          filepath='connections_gif', filename='globe_conn',
                          ortho_trans=None, daynight=True, **kwargs):
    '''
    Animates the network on a globe over time with an optional shadow for nighttime.

    Parameters
    ----------
    adj_mat_ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.
        3-dimensional Dataset whose coordinates are first_st, second_st, win_start.
    filepath : str, optional
        File path for storing the image files and gif. Default is
        'connections_gif' folder to be made in the current working directory.
    filename : str, optional
        File name for the gif, without file extension. Default is 'globe_conn'.
    ortho_trans : tuple, optional
        Orientation of the plotted globe; determines at what angle we view the globe.
        Defaults to average location of all stations.
    daynight : bool, optional
        Whether or not to include a shadow for nighttime. Default is True.

    Returns
    -------
    .png
        png image files used to make the gif animation, saved in filepath/images_for_giffing.
    .gif
        gif animation of the png image files, saved in filepath/gif.
    '''

    # check filepaths
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    im_filepath = filepath + '/images_for_giffing'
    if not os.path.exists(im_filepath):
        os.makedirs(im_filepath)

    # check filename
    if '.' in filename:
        if len(filename) > 4:
            filename = filename[:-4] # remove file extension
        else:
            print('Error: please input filename without file extension')
            return 'Error: please input filename without file extension'

    # get contstants
    list_of_stations = adj_mat_ds.first_st.values
    list_of_win_start = adj_mat_ds.win_start.values
    num_win = len(list_of_win_start)

    # check ortho_trans
    if ortho_trans is None:
        ortho_trans = svs.auto_ortho(list_of_stations)

    # initialize the list of names of image files
    names = []

    # plot the connections for each win_start value in the adjacency matrix
    for i in range(num_win):
        fig = svs.plot_connections_globe(adj_matrix = adj_mat_ds[dict(win_start = i)].adj_coeffs.values,
                                         list_of_stations = list_of_stations,
                                         time = list_of_win_start[i],
                                         ortho_trans = ortho_trans,
                                         daynight = daynight,
                                         **kwargs)
        im_name = im_filepath + '/%s.png' %i
        fig.savefig(im_name) # save image file
        names.append(im_name) # add name of image file to list

    # append plots to each other
    images = []
    for n in names:
        images.append(Image.open(n))

    # make gif file and save it in filepath
    images[0].save(filepath + '/%s.gif' %filename,
                   save_all = True,
                   append_images = images[1:],
                   duration = 50, loop = 0)
