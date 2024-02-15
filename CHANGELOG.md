v0.2.1 28.09.2023

- Removed Liquid folder and distributed its contents to other more appropriate submodules
- Deleted all non-liquid tasks for which an equivalent liquid task existed
- Updated tasks and methods that could use liquid dependencies but did not yet
- Updated the liquid engine base class to query and inform the agent when called outside an workflow class
- Jupyter and Colab notebooks can now be automatically built using a recipe based system
- Refactored and improved github actions
- Minor documentation improvements
- Miscellaneous bugfixes

v0.3.1 15.02.2023

- Refactoring metaprogramming. Now it uses the mako templating engine. c2cl is now a script that is used inside mako templates. tag2tag was removed.
- Added non local means denoising to the liquid engine in two different flavors that differ in the parallelization strategy.
- Major documentation improvements.
- Adding backend support for transonic, cupy and dask runtypes.
- Implementing numba, transonic, cupy and dask runtypes for 2D convolution
- Other QoL and miscellaneous bugfixing.

