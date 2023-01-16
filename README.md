# NanoPyx

[![License](https://img.shields.io/pypi/l/nanopyx.svg?color=green)](https://github.com/HenriquesLab/NanoPyx/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nanopyx.svg?color=green)](https://pypi.org/project/nanopyx)
[![Python Version](https://img.shields.io/pypi/pyversions/nanopyx.svg?color=green)](https://python.org)
[![tests](https://github.com/HenriquesLab/NanoPyx/workflows/tests/badge.svg)](https://github.com/HenriquesLab/NanoPyx/actions)
[![codecov](https://codecov.io/gh/HenriquesLab/NanoPyx/branch/main/graph/badge.svg)](https://codecov.io/gh/HenriquesLab/NanoPyx)

Nanoscopy Python library (NanoPyx, the successor to NanoJ) - focused on light microscopy and super-resolution imaging

---

## What is the NanoPyx 🔬 Library?

NanoPyx is a library specialized in the analysis of light microscopy and super-resolution data.
It is a successor to NanoJ, which is a Java library for the analysis of super-resolution microscopy data.

NanoPyx focuses on performance, by heavily exploiting cython aided multiprocessing and simplicity. It implements methods for the bioimage analysis field, with a special emphasis on those developed by the [Henriques Laboratory](https://henriqueslab.github.io/).

Currently it implements the following approaches:

- A reimplementation of the NanoJ drift correction
- ...

## Installation

You can install `NanoPyx` via [pip]:

```shell
pip install nanopyx
```

or if you want to install with all optional dependencies

```shell
pip install nanopyx[all]
```

To install latest development version :

    pip install git+https://github.com/HenriquesLab/NanoPyx.git

### Notes for Mac users

If you wish to compile the NanoPyx library from source, you will need to install the following dependencies:

- Homebrew from https://brew.sh/
- gcc, llvm and libomp from Homebrew through the command:

```shell
brew install gcc llvm libomp
```

## Run in jupyterlab within a docker container

```shell
docker run --name nanopyx1 -p 8888:8888 henriqueslab/nanopyx:latest
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v2.0] license,
"NanoPyx" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[gnu gpl v2.0]: http://www.gnu.org/licenses/gpl-2.0.txt
[apache software license 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[mozilla public license 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[file an issue]: https://github.com/HenriquesLab/NanoPyx/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[pypi]: https://pypi.org/
