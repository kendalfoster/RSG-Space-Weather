## Packages
import matplotlib.pyplot as plt # maybe can delete this
import xarray as xr # if gives error, just rerun


def plot_mag_data(ds):
    """
    Plot components of data like on the SuperMAG website.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.

    Yields
    -------
    NoneType
        One column of plots, where each plot shows the components of one station over time.

    """
    ds.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1)
