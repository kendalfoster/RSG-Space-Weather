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
