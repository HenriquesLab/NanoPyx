r"""

# What is the 🔬 NanoPyx Library?

NanoPyx is a library specialized in the analysis of light microscopy and super-resolution data.
It is a successor to NanoJ, which is a Java library for the analysis of super-resolution microscopy data.

NanoPyx focuses on performance, by heavily exploiting cython aided multiprocessing and simplicity. It 
implements methods for the bioimage analysis field, with a special emphasis on those developed by the 
[Henriques Laboratory](https://henriqueslab.github.io/).

Currently it implements the following approaches:
 - A reimplementation of the NanoJ drift correction
 - ...

# Quickstart

## Install from PyPI
```shell
pip install nanopyx
```
or if you want to install with all optional dependencies
```shell
pip install nanopyx[all]
```

## Install from source
```shell
pip install git+https://github.com/HenriquesLab/NanoPyx.git
```

### Notes for Mac users
If you wish to compile the NanoPyx library from source, you will need to install the following dependencies:
- Homebrew from https://brew.sh/
- gcc, llvm and libomp from Homebrew
```shell
brew install gcc llvm libomp
```

## Run in jupyterlab within a docker container
```shell
docker run --name nanopyx1 -p 8888:8888 henriqueslab/nanopyx:latest
```

"""
