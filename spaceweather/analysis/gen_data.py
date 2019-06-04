"""
Contents
--------

- generate_one_day_one_component_time_series
- generate_one_day_time_series
"""


import numpy as np
from numpy import hstack
import scipy as sp
from scipy import signal
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt

# This function generates a time series (with no_of_components components) for a full day
# At some point during the day there will be Pc waves
# We input the start date and start time of the Pc waves; we generate data for the (full) start_date day
# The charateristics of the Pc waves are set by the inputs wavepacket_duration (in mins) and number_of_waves

# For generating underlying data we use an OU process
# Code comes from https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/

# phase_shift should be in the range [0,1]
# it specifies how much to offset the waves by, as a proportion of the full wave cycle

def generate_one_day_one_component_time_series(pc_wave_start_date, pc_wave_start_time, wavepacket_duration, number_of_waves, phase_shift = 0):

    date_time = pd.to_datetime(pc_wave_start_date + ' ' + pc_wave_start_time)
    total_timesteps = int(np.timedelta64(1,'D')/np.timedelta64(1,'m'))
    full_day_timeseries = np.zeros(total_timesteps)
    data_source = ['' for i in range(total_timesteps)]

    # first generate the wavepacket - a sine wave combined with a Gaussian window
    gaussian_window = signal.gaussian(wavepacket_duration+1, std=(wavepacket_duration+1)/6)
    sine_wave = np.zeros(wavepacket_duration+1)

    for minute in range(wavepacket_duration+1):
        sine_wave[minute] = np.sin((minute - phase_shift * wavepacket_duration/number_of_waves) * (2 * np.pi) * number_of_waves/wavepacket_duration)

    wavepacket_start_index = int((date_time-pd.to_datetime(pc_wave_start_date))/np.timedelta64(1,'m'))
    for i in range(wavepacket_duration+1):
        full_day_timeseries[wavepacket_start_index+i] = gaussian_window[i] * sine_wave[i] * 100
        data_source[wavepacket_start_index+i] = 'wavepacket'

    # next generate some random behaviour before and after the wavepacket
    # use an Ornstein-Uhlenbeck process (rewritten as a Langevin equation) to generate the other noisy data

    # first define the parameters
    # adjust sigma and tau to change the shape of the wavepacket
    sigma = 38  # Standard deviation. From (a single) empirical observation
    mu = 0.  # Mean.
    dt = 1.  # Time step.
    tau = 50. * dt # Time constant. This choice seems to yield reasonable-looking results
    T = 1440.  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.

    # things that are used in the formulae
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)

    # first complete the time series by populating the timesteps before the wavepacket
    # note that we use the time-reversibility property of the O-U process
    start_index_start = 0
    end_index_start = wavepacket_start_index

    # add 1 so that there is an overlap (of 1 timestep) between the OU process and the wavepacket
    # the first datapoint of the wavepacket will be used as the first datapoint of the O-U process
    first_part = np.zeros(end_index_start-start_index_start + 1)
    first_part[0] = full_day_timeseries[wavepacket_start_index]

    # populate the first part of the O-U process (before the wavepacket)
    for i in range(len(first_part) - 1):
        first_part[i + 1] = first_part[i] + dt * (-(first_part[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

    for i in range(len(first_part)):
        index = end_index_start - i
        full_day_timeseries[index] = first_part[i]
        if data_source[index] == 'wavepacket' and index != wavepacket_start_index:
            print('duplicate')
        elif data_source[index] == 'wavepacket' and index == wavepacket_start_index:
            data_source[index] = 'overlap'
        else:
            data_source[index] = 'OU_first_part'

    # now populate the last part of the O-U process (after the wavepacket)
    # note start_index_start, end_index_start, start_index_last and end_index_last are all array indices, hence the -1 in end_index_last
    start_index_last = wavepacket_start_index + wavepacket_duration
    end_index_last = int(np.timedelta64(1,'D')/np.timedelta64(1,'m')) - 1

    last_part = np.zeros(end_index_last - start_index_last + 1)
    last_part[0] = full_day_timeseries[start_index_last]

    # populate the last part of the O-U process (after the wavepacket)
    for i in range(len(last_part) - 1):
        last_part[i + 1] = last_part[i] + dt * (-(last_part[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

    for i in range(len(last_part)):
        index = start_index_last + i
        full_day_timeseries[index] = last_part[i]
        if (data_source[index] == 'wavepacket' or data_source[index] == 'OU_first_part') and index != start_index_last:
            print(index)
            print('duplicate')
        elif data_source[index] == 'wavepacket' and index == start_index_last:
            data_source[index] = 'overlap'
        else:
            data_source[index] = 'OU_last_part'

    return full_day_timeseries

def generate_one_day_time_series(pc_wave_start_date, pc_wave_start_time, wavepacket_duration, number_of_waves, phase_shift = [0, np.random.rand(), np.random.rand()], station = ['XXX']):
    day_start_time = pd.to_datetime(pc_wave_start_date)
    day_end_time = pd.to_datetime(pc_wave_start_date) + np.timedelta64(1,'D') - np.timedelta64(1,'m')
    total_timesteps = int(np.timedelta64(1,'D')/np.timedelta64(1,'m'))

    times = pd.to_datetime(np.linspace(day_start_time.value, day_end_time.value, total_timesteps))

    components = ['N', 'E', 'Z']
    measurement_data = np.zeros((len(times),len(components),len(station)))

    for station_index in range(len(station)):



        if station_index == 0:
            N_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:00:00', 30, 4, phase_shift = phase_shift[0])
            E_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:00:00', 30, 4, phase_shift = phase_shift[1])
            Z_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:00:00', 30, 4, phase_shift = phase_shift[2])
        else:
            N_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:06:00', 30, 4, phase_shift = phase_shift[0])
            E_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:06:00', 30, 4, phase_shift = phase_shift[1])
            Z_component_time_series = generate_one_day_one_component_time_series('2001-04-03', '08:06:00', 30, 4, phase_shift = phase_shift[2])



        measurement_data[:,0,station_index] = N_component_time_series
        measurement_data[:,1,station_index] = E_component_time_series
        measurement_data[:,2,station_index] = Z_component_time_series

    dataarray = xr.DataArray(data = measurement_data,
                  coords = [times, components, station],
                  dims = ['time', 'component', 'station'])

    dataset = xr.Dataset(data_vars = {'measurements': (['time', 'component', 'station'], dataarray)},
                    coords = {'time': times,
                              'component': components,
                              'station': station})

    return dataset
