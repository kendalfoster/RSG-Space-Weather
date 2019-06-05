"""
Contents
--------

- plot_mag_data
- plot_cca_ang
"""


## Packages
import matplotlib.pyplot as plt # maybe can delete this
import xarray as xr # if gives error, just rerun


def plot_mag_data(ds):
    """
    Plot components of data like on the SuperMAG website.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`spaceweather.analysis.data_funcs.csv_to_Dataset`.

    Yields
    -------
    matplotlib.figure.Figure
        One column of plots, where each plot shows the components of one station over time.
    """

    fig = plt.figure(figsize = (16,4))
    ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)

    return fig


def plot_cca_ang_pair(cca_ang_ds):
    """
    Plot CCA angles of data.

    Parameters
    ----------
    ds : xarray.Dataset
        Output from :func:`spaceweather.analysis.cca.cca_anges`.

    Yields
    -------
    matplotlib.figure.Figure
        Plot showing the angle relative to the data for each weight.
    """

    tit = 'Station Pair: ' + cca_ang_ds.first_st.values.flatten()[0] + ' & ' + cca_ang_ds.second_st.values.flatten()[0]

    fig = plt.figure(figsize = (16,4))
    cca_ang_ds.ang_data.plot.line(x='time', hue='a_b')
    plt.title(tit, fontsize = 30)
    plt.ylim(0, 180)
    plt.ylabel('Angle', fontsize = 20)
    plt.xlabel('Time', fontsize = 20)

    return fig
