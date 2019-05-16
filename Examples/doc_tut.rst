======================
Documentation Tutorial
======================

We are using the Numpy standard of documentation in our functions. The style guidelines can be found at https://sphinxcontrib-napoleon.readthedocs.io/en/latest/. Essentially just follow the template, capitalize the beginnings of phrases, and put a period at the end of each statement- even if they're not full sentences.
We are using Sphinx to build the documentation. You can get Sphinx at http://www.sphinx-doc.org/en/master/usage/installation.html.

Write the Docs
--------------
Basically, inside the function definition you include the documentation sandwiched between triple quotes """.

- Note the triple quotes each go on their own lines, without a blank line in between the actual documentation.
- Also note that there is a blank line between the sections of the documentation; without this, the documentation descends into chaos and gets mucky.
- If you want something a description of something to have a line break, include "\n".
- To reference another function in the documentation, use ":func:`path_to_and_name_of_other_function`". Note the `, not a '.
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
