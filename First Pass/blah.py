# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import lib.supermagyz as yz
# import lib.supermag as sm
# from matplotlib.animation import FuncAnimation

from PIL import Image



ortho_trans = (-100, 40)


def data_globe_gif(time_start = 0, time_end = 10, ortho_trans = (0, 0), file_name = "test"):
    #times in terms of index in the array, might be helpful to have a fn to look up index from timestamps
    names = []
    images = []

    for i in range(time_start, time_end):
        t = station_readings.time[i]
        fig = yz.plot_data_globe(station_readings, t, list_of_stations, ortho_trans)
        fig.savefig("gif/images_for_giffing/%s.png" %i)

    for i in range(time_start, time_end):
        names.append("gif/images_for_giffing/%s.png" %i)

    for n in names:
        frame = Image.open(n)
        images.append(frame)

    images[0].save("gif/%s.gif" %file_name, save_all = True, append_images = images[1:], duration = 50, loop = 0)

data_globe_gif(ortho_trans = (-100, 40))
