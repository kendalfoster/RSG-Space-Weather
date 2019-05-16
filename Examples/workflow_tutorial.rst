=================
Workflow Tutorial
=================

The master branch hosts the "official" progress of the project, with only code that has been confirmed to work properly. Each individual has a personal branch that is used for code developing and testing. Once the individual's code is production-ready, it can be merged with the master branch; this is how the master branch should be developed. After a merge- or at least a significant merge- it is often a good idea for the individual branches to be "rebased" from the master branch. This process takes the changes to the master branch and applies them to the individual branch, thus updating the code on the individual branch to the most recent production-ready version.




Merge the individual branch with master
---------------------------------------
Once the code on the individual branch is confirmed to work properly and is ready to be merged with the master branch, a pull request must be opened on GitHub.

- go to the repository page on the GitHub website
- next to the button that switches branches, click the button labeled "New pull request"
- in the first dropdown menu labeled "base:" ensure the master branch is selected
- in the second dropdown menu labeled "compare:" ensure the individual branch is selected
- assuming the merge is able to happen, write a descriptive title for the merge and possibly a longer comment if necessary
- scroll down and click the big green button labeled "Create pull request"
- deal with any conflicts that may arise (this can be painful, very painful)
- finally, scroll down and click the big green button labeled "Merge pull request"




Rebase the individual branch from master
----------------------------------------
After a merge or perhaps a significant merge, it is often useful to rebase the individual branch to update code to the most recent version. This is done via the command line/terminal.

- open the command line/terminal and navigate to the git repository folder
- check which branch you are currently on with::

  git branch -a
  
- then do this
