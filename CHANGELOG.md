v0.2.1 28.09.2023

- Removed Liquid folder and distributed its contents to other more appropriate submodules
- Deleted all non-liquid tasks for which an equivalent liquid task existed
- Updated tasks and methods that could use liquid dependencies but did not yet
- Updated the liquid engine base class to query and inform the agent when called outside an workflow class
- Jupyter and Colab notebooks can now be automatically built using a recipe based system
- Refactored and improved github actions
- Minor documentation improvements
- Miscellaneous bugfixes