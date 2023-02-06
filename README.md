# Under Development, currently in alpha stage

Implementations of drift alignment and channel registration should be working, remaining features are under development and/or broken (for now)

# NanoPyx

[![License](https://img.shields.io/pypi/l/nanopyx.svg?color=green)](https://github.com/HenriquesLab/NanoPyx/branch/main/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/nanopyx.svg?color=green)](https://pypi.org/project/nanopyx)
[![Python Version](https://img.shields.io/pypi/pyversions/nanopyx.svg?color=green)](https://python.org)
[![Tests](https://github.com/HenriquesLab/NanoPyx/actions/workflows/test_package.yml/badge.svg)](https://github.com/HenriquesLab/NanoPyx/actions/workflows/python-package-test.yml)
[![codecov](https://codecov.io/gh/HenriquesLab/NanoPyx/branch/main/graph/badge.svg)](https://codecov.io/gh/HenriquesLab/NanoPyx)
[![Downloads](https://img.shields.io/pypi/dm/nanopyx)](https://pypi.org/project/nanopyx)
[![Docs](https://img.shields.io/badge/documentation-link-blueviolet)](https://henriqueslab.github.io/NanoPyx)
[![Build status](https://ci.appveyor.com/api/projects/status/oc7pk3t2h04r60j4?svg=true)](https://ci.appveyor.com/project/paxcalpt/nanopyx)

Nanoscopy Python library (NanoPyx, the successor to NanoJ) - focused on light microscopy and super-resolution imaging

---

## What is the NanoPyx 🔬 Library?

NanoPyx is a library specialized in the analysis of light microscopy and super-resolution data.
It is a successor to [NanoJ](https://github.com/HenriquesLab/NanoJ-Core), which is a Java library for the analysis of super-resolution microscopy data.

NanoPyx focuses on performance, by heavily exploiting cython aided multiprocessing and simplicity. It implements methods for the bioimage analysis field, with a special emphasis on those developed by the [Henriques Laboratory](https://henriqueslab.github.io/).

Currently it implements the following approaches:

- A reimplementation of the NanoJ drift correction and channel registration methods
- More to come soon™

## Installation

You can install `NanoPyx` via [pip]:

```shell
pip install nanopyx
```

or if you want to install with all optional dependencies

```shell
pip install 'nanopyx[all]'
```

To install latest development version :

```shell
pip install git+https://github.com/HenriquesLab/NanoPyx.git
```

### Notes for Mac users

If you wish to compile the NanoPyx library from source, you will need to install the following dependencies:

- Homebrew from <https://brew.sh/>
- gcc, llvm and libomp from Homebrew through the command:

```shell
brew install gcc llvm libomp
```

## Run in jupyterlab within a docker container

```shell
docker run --name nanopyx1 -p 8888:8888 henriqueslab/nanopyx:latest
```

## Contributing

Contributions are very welcome.  
Please read our [Contribution Guidelines](https://github.com/HenriquesLab/NanoPyx/blob/main/CONTRIBUTING.md) to know how to proceed.

## License

Distributed under the terms of the [GNU GPL v2.0] license,
"NanoPyx" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[gnu gpl v2.0]: http://www.gnu.org/licenses/gpl-2.0.txt
[file an issue]: https://github.com/HenriquesLab/NanoPyx/issues
[pip]: https://pypi.org/project/pip/

## Development at a glance

## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Structure

[![Structure](https://images.repography.com/33651790/HenriquesLab/NanoPyx/structure/6USKh-PjgkYlbiepDRN9aThOShl3TNx_VkIycH0M6e0/Sqp8CSmE3HObh4_sa8_-IsUByYshpCVQpMuu1E_Fwiw_table.svg)](https://github.com/HenriquesLab/NanoPyx)
