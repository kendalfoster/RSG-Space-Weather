## Packages
import numpy as np
import matplotlib.pyplot as plt
import spaceweather.visualisation.globes as svg
from PIL import Image
# Local Packages
# import spaceweather.rcca as rcca


def data_globe_gif(station_readings, time_start = 0, time_end = 10, ortho_trans = (0, 0), colour = True, file_name = "sandra"):
    #times in terms of index in the array, might be helpful to have a fn to look up index from timestamps
    names = []
    images = []
    list_of_stations = station_readings.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = svg.auto_ortho(list_of_stations)

    if colour:
        for i in range(time_start, time_end):
            t = station_readings.time[i].data
            fig = svg.plot_data_globe_colour(station_readings, t, list_of_stations, ortho_trans)
            fig.savefig("Scratch (Tinkerbell)/gif/images_for_giffing/%s.png" %i)
    else:
        for i in range(time_start, time_end):
            t = station_readings.time[i]
            fig = svg.plot_data_globe(station_readings, t, list_of_stations, ortho_trans)
            fig.savefig("Scratch (Tinkerbell)/gif/images_for_giffing/%s.png" %i)

    for i in range(time_start, time_end):
        names.append("Scratch (Tinkerbell)/gif/images_for_giffing/%s.png" %i)

    for n in names:
        frame = Image.open(n)
        images.append(frame)

    images[0].save("Scratch (Tinkerbell)/gif/%s.gif" %file_name, save_all = True, append_images = images[1:], duration = 50, loop = 0)


##
def data_globe_gif_colour(station_readings, list_of_stations, time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "sandra"):
    #times in terms of index in the array, might be helpful to have a fn to look up index from timestamps
    names = []
    images = []
    # list_of_stations = station_readings.station
    if np.all(ortho_trans == (0, 0)):
        ortho_trans = svg.auto_ortho(list_of_stations)

    for i in range(time_start, time_end):
        t = station_readings.time[i]
        fig = plot_data_globe(station_readings, t, list_of_stations, ortho_trans)
        fig.savefig("gif/images_for_giffing/%s.png" %i)

    for i in range(time_start, time_end):
        names.append("gif/images_for_giffing/%s.png" %i)

    for n in names:
        frame = Image.open(n)
        images.append(frame)

    images[0].save("gif/%s.gif" %file_name, save_all = True, append_images = images[1:], duration = 50, loop = 0)
