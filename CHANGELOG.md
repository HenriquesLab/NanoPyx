# Changelog

All notable changes to NanoPyx will be documented in this file.

## v1.3.2 - 2025-11-07

- Added `pad_edges` option to use edge padding instead of leaving margins as 0 in SRRF/eSRRF
- Minor bug fixes and improvements

## v1.3.1 - 2025-10-28

- Implemented FHT (Fast Hankel Transform) interpolation for eSRRF calculation
- Added two-point gradient calculation implementation
- Removed redundant double gradient interpolation from eSRRF workflow
- Added Liquid Engine class for gradient calculations
- Improved performance and stability of gradient computation
- Fixed CI/CD issues with dependency management (dask, pyarrow)
- Added manylinux dependencies for better compatibility

## v1.3.0 - 2025-09-11

- Major refactoring of eSRRF gradient interpolation methods
- Improved FHT interpolation implementation
- Enhanced testing infrastructure and removed problematic tests
- Updated dependencies and fixed compatibility issues
- Improved CI/CD pipeline stability

## v1.2.3 - 2025-07-09

- Fixed frames_per_timepoint handling in eSRRF 3D
- Improved radius_z calculation from PSF_ratio and voxel_ratio in eSRRF 3D
- Fixed issue with limit on number of frames
- Added explicit OpenCL library support in CI/CD
- Changed default grad_magnification to 2 for better performance
- Added frames_per_timepoint and temporal_correlation parameters to eSRRF workflow
- Updated notebooks to reflect new eSRRF workflow changes
- Fixed output type enforcement to np.ndarray
- Improved build system compatibility with newer manylinux images

## v1.1.0 - 2025-04-10

- Major eSRRF update with 3D support improvements
- Updated channel registration to use scipy gaussian filtering
- Fixed channel registration gaussian blur to use 2D kernel matching new convolution class
- Added verbose option to drift and channel registration
- Fixed Robert's cross implementation in eSRRF CPU runtypes
- Made publishing to PyPI automatic
- Updated eSRRF radius and edges calculation
- Fixed angle rotation for modulo-2 kernels in convolution
- Added support for 3D arrays in dask and transonic implementations
- Updated Convolution2D to accept and properly handle 3D arrays
- Fixed index clamping in edge cases
- Improved macOS wheel building for different architectures

## v1.0.0 - 2025-01-07

- First stable major release
- Comprehensive testing and validation across all platforms
- Stabilized API and workflows

## v0.6.1 - 2024-07-12

- Added Python 3.12 support
- Minor bug fixes to image decorrelation
- Isolated NLM benchmarking notebook
- Updated sigma parameters
- Improved pytest coverage handling
- Bug fixes and stability improvements

## v0.6.0 - 2024-07-05

- Extended fuzzy logic showcase
- Added Python 3.12 support across all platforms
- Added codecov token for coverage uploads
- Added coverage badge to repository
- Improved RCC (Radial Cross Correlation) testing
- Better handling of systems without OpenCL available during testing
- Added default bead image to load example data
- Implemented basic 3D eSRRF functionality
- Fixed 3D temporal correlation frame counting
- Updated benchmarking with new parameters
- Added benchmarking notebook for different hardware configurations
- Improved handling of single OpenCL device scenarios
- Downgraded macOS x86 version for better compatibility

## v0.5.0 - 2024-06-05

- Major version bump with stabilized features
- Comprehensive testing improvements
- Enhanced CI/CD pipeline

## v0.4.0 - 2024-03-08

- Major testing infrastructure improvements
- Fixed Lanczos interpolation issues
- Improved parallel testing with pytest
- Added more comprehensive benchmarking tests
- Enhanced channel registration testing
- Disabled problematic tests and improved test reliability
- Fixed imports and type inference in compilation
- Improved CI/CD workflow stability

## v0.3.1 - 2024-01-24

- Refactoring metaprogramming using Mako templating engine
- c2cl is now a script used inside Mako templates; tag2tag was removed
- Added Non-Local Means (NLM) denoising to the Liquid Engine in two flavors with different parallelization strategies
- Major documentation improvements
- Added backend support for transonic, cupy, and dask runtypes
- Implemented numba, transonic, cupy, and dask runtypes for 2D convolution
- Fixed OpenCL NLM denoising bugs (fast and non-fast implementations)
- Added better memory management for OpenCL operations
- Improved handling of edge cases in buffer size calculations
- Fixed type casting issues for integrated Intel GPUs
- Better handling of memory issues with OpenCL
- QoL improvements and miscellaneous bug fixes

## v0.2.2 - 2023-10-10

- Updated build system to use cibuildwheels
- Improved wheel building workflows for multiple platforms
- Added polar transform implementation with OpenCL support
- Fixed polar transform bugs in OpenCL kernels
- Added pytest for polar transforms with image comparison via PCC
- Updated dependencies and requirements
- Fixed widget rendering in notebooks
- Improved nox configuration
- Better handling of missing requirements
- PNG format converted to JPEG for PIL compatibility
- Bug fixes in ARM wheel building

## v0.2.1 - 2023-09-29

- Removed Liquid folder and distributed its contents to other more appropriate submodules
- Deleted all non-liquid tasks for which an equivalent liquid task existed
- Updated tasks and methods that could use liquid dependencies but did not yet
- Updated the Liquid Engine base class to query and inform the agent when called outside a workflow class
- Jupyter and Colab notebooks can now be automatically built using a recipe-based system
- Refactored and improved GitHub Actions
- Minor documentation improvements
- Miscellaneous bug fixes

## v0.2.0 - 2023-08-14

- Initial structured release
- Core functionality established

## v0.1.2 - 2023-05-09

- Bug fixes and improvements

## v0.1.1 - 2023-05-09

- Minor updates and fixes

## v0.1.0 - 2023-05-09

- Initial public release
- Basic SRRF/eSRRF implementation
- Drift correction functionality
- Channel alignment tools
- OpenCL acceleration support

## v0.0.1 - 2023-03-04

- Initial development release
- Project structure established
- Core architecture implemented

