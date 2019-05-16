## Packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import xarray as xr # if gives error, just rerun
# Local Packages
import spaceweather.rcca as rcca

def power_spectrum(ts=None, ds=None, station=None, component=None):
    """
    Plot the power spectrum of the Fourier transform of the time series.
    It is recommended to use a small amount of the time series for best results.

    Parameters
    ----------
    ts : xarray.Dataset, optional
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        Can be replaced by including ds, station, and component inputs.
        Timeseries of one component in one station.
    ds : xarray.Dataset, optional
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This will be used to extract a timeseries of the same form as the ts input.
    station : string, optional
        Three letter code for the station to be used in the extraction of timeseries from ds input.
    component : string, optional
        Component to be used in the extraction of timeseries from ds input.

    Yields
    -------
    matplotlib.figure.Figure
        Plot of the power spectrum. This may be a list of lines?
    """

    # prepare the time series
    if ts is None:
        ts = ds.loc[dict(station = station, component = component)]
    ts = ts.dropna(dim = 'time', how = 'any').measurements

    # fast Fourier transform the time series
    ft_ts = sp.fftpack.fft(ts)

    # get the frequencies corresponding to the FFT
    n = len(ts)
    freqs = sp.fftpack.fftfreq(n = n, d = 1)

    # plot power spectrum
    fig = plt.figure(figsize=(10,8))
    plt.plot(freqs[0:n//2], np.abs(ft_ts[0:n//2]))
    plt.title('Power Spectrum', fontsize=30)
    plt.xlabel('Frequency, cycles/min', fontsize=20)
    plt.ylabel('Intensity, counts', fontsize=20)
    # explicitly returning the figure results in two figures being shown


def spectrogram(ts=None, ds=None, station=None, component = None, win_len=128, win_olap=None):
    """
    Plot a spectrogram for one component of one station.

    Parameters
    ----------
    ts : xarray.Dataset, optional
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        Can be replaced by including ds, station, and component inputs.
        Timeseries of one component in one station.
    ds : xarray.Dataset, optional
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
        This will be used to extract a timeseries of the same form as the ts input.
    station : string, optional
        Three letter code for the station to be used in the extraction of timeseries from ds input.
    component : string, optional
        Component to be used in the extraction of timeseries from ds input.
    win_len : int, optional
        Length of the window. Default is 128 (minutes).
    win_olap : int, optional
        Length of the overlap of consecutive windows. Default is win_len - 1 for
        a new window every minute.

    Yields
    -------
    matplotlib.figure.Figure
        Plot of the spectrogram. This may be a NoneType?
    """

    # prepare the time series
    if ts is None:
        ts = ds.loc[dict(station = station, component = component)]
    ts = ts.dropna(dim = 'time', how = 'any').measurements

    # check the window overlap
    if win_olap is None:
        win_olap = win_len - 1

    # setup spectrogram
    f, t, Sxx = sp.signal.spectrogram(ts, nperseg = win_len, noverlap = win_olap)

    # plot spectrogram
    fig = plt.figure(figsize=(10,8))
    plt.pcolormesh(t, f, Sxx, norm = colors.LogNorm(vmin = 1, vmax = 20000))
    plt.title('Spectrogram', fontsize=30)
    plt.xlabel('Time Window', fontsize=20)
    plt.ylabel('Frequency, cycles/min', fontsize=20)
    plt.colorbar(label='Intensity')
    fig.axes[-1].yaxis.label.set_size(20)
    # explicitly returning the figure results in two figures being shown
