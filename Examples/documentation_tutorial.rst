======================
Documentation Tutorial
======================

We are using the Numpy standard of documentation in our functions. The style guidelines can be found here [insert url]. Essentially just follow the template, capitalize the beginnings of phrases, and put a period at the end of each statement- even if they're not full sentences.
We are using Sphinx to build the documentation. You can get Sphinx here [insert url].

Write the Docs
--------------
Basically, inside the function definition you include the documentation sandwiched between triple quotes """.

- Note the triple quotes each go on their own lines, without a blank line in between the actual documentation.
- Also note that there is a blank line between the sections of the documentation; without this, the documentation descends into chaos and gets mucky.
- If you want something a description of something to have a line break, include "\n".
- To reference another function in the documentation, use ":func:`path_to_and_name_of_other_function`".
- To denote a parameter as optional, add ", optional" after its type.

Here is a template::
  def function(param1, param2=5):
      """
      A short description of the function.

      A longer description of the function, maybe include more details or relation to other functions. You could also explain why the function exists, or you could just write words to fill the space. Please don't do that; not every function requires a longer description.

      Parameters
      ----------
      param1 : type_of_param1
          A short description of param1.
      param2 : type_of_param2, optional
          A short description of param2.

      Returns
      -------
      type_of_return_object
          A short description of the return object.
              A longer description of the return object.\n
              I found it helpful to occasionally have new lines.
      """

      nonsense = param1 + param2
      something = np.cos(nonsense)
      notadataset = something/2

      return notadataset

Here is an example from our code (with the function shortened)::
def mag_detrend(ds, type='linear'):
    """
    Detrend the time series for each component.

    Parameters
    ----------
    ds : xarray.Dataset
        Data as converted by :func:`supermag.mag_csv_to_Dataset`.
    type : str, optional
        Type of detrending passed to scipy detrend. Default is 'linear'.

    Returns
    -------
    xarray.Dataset
        Dataset with the SuperMAG data easily accessible.
            The data_vars are: measurements.\n
            The coordinates are: time, component, station.
    """

    stations = ds.station
    components = ds.component

    for i in range(1, len(stations)):
        da = xr.concat([da, temp_da], dim = 'station')

    res = da.to_dataset(name = 'measurements')

    return res





Build the Docs
--------------
This is the part that uses Sphinx, so make sure you have it installed.

- Sphinx is a command line/terminal tool, so open up the terminal and navigate to the `docs` folder in the git repository.
- Everything should be set up- all you need to do is run the command `make html`.
- You need to rerun this command each time you want to update the documentation.
- There is a way to host the documentation on readthedocs and have it update automatically with GitHub commits, but save that for later.

You can view the documentation by navigating to::
  git repo > docs > build > html > index.html
