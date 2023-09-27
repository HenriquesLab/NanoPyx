![Logo](https://user-images.githubusercontent.com/7071808/259983552-ef7aa814-16ba-4947-a69f-1f096d70370e.png)
# Actively being developed with stable releases
[![PyPI](https://img.shields.io/pypi/v/nanopyx.svg?color=green)](https://pypi.org/project/nanopyx)
[![Python Version](https://img.shields.io/pypi/pyversions/nanopyx.svg?color=green)](https://python.org)
[![Downloads](https://img.shields.io/pypi/dm/nanopyx)](https://pypi.org/project/nanopyx)
[![Docs](https://img.shields.io/badge/documentation-link-blueviolet)](https://henriqueslab.github.io/NanoPyx)
[![License](https://img.shields.io/github/license/HenriquesLab/NanoPyx?color=Green)](https://github.com/HenriquesLab/NanoPyx/blob/main/LICENSE.txt)
[![Tests](https://github.com/HenriquesLab/NanoPyx/actions/workflows/nanopyx_oncall_mechanic.yml/badge.svg)](https://github.com/HenriquesLab/NanoPyx/actions/workflows/nanopyx_oncall_mechanic.yml)
[![Contributors](https://img.shields.io/github/contributors-anon/HenriquesLab/NanoPyx)](https://github.com/HenriquesLab/NanoPyx/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/HenriquesLab/NanoPyx?style=social)](https://github.com/HenriquesLab/NanoPyx/)
[![GitHub forks](https://img.shields.io/github/forks/HenriquesLab/NanoPyx?style=social)](https://github.com/HenriquesLab/NanoPyx/)
[![DOI](https://zenodo.org/badge/505388398.svg)](https://zenodo.org/badge/latestdoi/505388398)

Nanoscopy Python library (NanoPyx, the successor to NanoJ) - focused on light microscopy and super-resolution imaging

---

## What is the NanoPyx 🔬 Library?

NanoPyx is a library specialized in the analysis of light microscopy and super-resolution data.
It is a successor to [NanoJ](https://github.com/HenriquesLab/NanoJ-Core), which is a Java library for the analysis of super-resolution microscopy data.

NanoPyx focuses on performance, by heavily exploiting cython aided multiprocessing and simplicity. It implements methods for the bioimage analysis field, with a special emphasis on those developed by the [Henriques Laboratory](https://henriqueslab.github.io/).
It will be distributed as a Python Library and also as [Codeless Jupyter Notebooks](https://github.com/HenriquesLab/NanoPyx#codeless-jupyter-notebooks-available), that can be run locally or on Google Colab, and as a [napari plugin](https://github.com/HenriquesLab/NanoPyx#napari-plugin).

You can read more about NanoPyx in our [preprint](https://www.biorxiv.org/content/10.1101/2023.08.13.553080v1).

Currently it implements the following approaches:
- A reimplementation of the NanoJ image registration, SRRF and Super Resolution metrics
- More to come soon™

if you found this work useful, please cite: [preprint](https://www.biorxiv.org/content/10.1101/2023.08.13.553080v1) and  [![DOI](https://zenodo.org/badge/505388398.svg)](https://zenodo.org/badge/latestdoi/505388398)

## Short Video Tutorials
| What is NanoPyx? | How to use NanoPyx in Google Colab? |
|:-:|:-:|
| [![](https://user-images.githubusercontent.com/7071808/259985020-b629a570-f131-4666-aadb-ba62ac7dbea2.png)](https://youtu.be/iAdgusBAU0Q) | [![](https://user-images.githubusercontent.com/7071808/259985779-4403d895-76a8-4050-bfd7-9317516a8f3e.png)](https://youtu.be/KD0RzolFnd4) |

## Codeless jupyter notebooks available:

| Category | Method | Last test | Notebook | Colab Link |
| --- | --- | --- | --- | --- |
| Registration | Channel Registration |  ✅ by ADB (13/08/23) | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-blue.svg?style=flat&logo=jupyter&logoColor=white)](https://github.com/HenriquesLab/NanoPyx/blob/main/notebooks/ChannelRegistration.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/HenriquesLab/NanoPyx/blob/main/notebooks/ChannelRegistration.ipynb) |
| Registration | Drift Correction | ✅ by ADB (13/08/23) | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-blue.svg?style=flat&logo=jupyter&logoColor=white)](https://github.com/HenriquesLab/NanoPyx/blob/main/notebooks/DriftCorrection.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/HenriquesLab/NanoPyx/blob/main/notebooks/DriftCorrection.ipynb) |
| Quality Control | Image fidelity and resolution metrics | ✅ by ADB (13/08/23) | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-blue.svg?style=flat&logo=jupyter&logoColor=white)](https://github.com/HenriquesLab/NanoPyx/blob/main/notebooks/SRMetrics.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/HenriquesLab/NanoPyx/blob/main/notebooks/SRMetrics.ipynb) |
| Super-resolution | SRRF | ✅ by ADB (13/08/23) | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-blue.svg?style=flat&logo=jupyter&logoColor=white)](https://github.com/HenriquesLab/NanoPyx/blob/main/notebooks/SRRFandQC.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/HenriquesLab/NanoPyx/blob/main/notebooks/SRRFandQC.ipynb) |
| Super-resolution | eSRRF | ✅ by BMS (11/08/23) | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-blue.svg?style=flat&logo=jupyter&logoColor=white)](https://github.com/HenriquesLab/NanoPyx/blob/main/notebooks/eSRRFandQC.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/HenriquesLab/NanoPyx/blob/main/notebooks/eSRRFandQC.ipynb) |
| Tutorial | Notebook with Example Dataset | ✅ by ADB (13/08/23) | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-blue.svg?style=fflat&logo=jupyter&logoColor=white)](https://github.com/HenriquesLab/NanoPyx/blob/main/notebooks/ExampleDataSRRFandQC.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/HenriquesLab/NanoPyx/blob/main/notebooks/NanoPyxExampleTutorial.ipynb) |

## napari plugin

NanoPyx is also available as a napari plugin, which can be installed via pip:

```pip install napari-nanopyx```

## Installation

`NanoPyx` is compatible and tested with Python 3.9, 3.10, 3.11 in MacOS, Windows and Linux. Installation time depends on your hardware and internet connection, but should take around 5 minutes.

You can install `NanoPyx` via [pip]:

```shell
pip install nanopyx
```

If you want to install with support for Jupyter notebooks:

```shell
pip install nanopyx[jupyter]
```

or if you want to install with all optional dependencies:

```shell
pip install nanopyx[all]
```

To install latest development version:

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

## Usage

Depending on your preferences and coding proficiency you might be using NanoPyx differently. 

- If you are using Jupyter Notebooks or Google Colab notebooks check out our [video tutorial](https://youtu.be/KD0RzolFnd4)
- If you are using our [napari plugin](https://github.com/HenriquesLab/NanoPyx#napari-plugin) check out the official [napari tutorial](https://napari.org/stable/tutorials/index.html) and stay tuned for more!
- If you prefer to use the Python library and take full advantage of the Liquid Engine flexibility, check out our Liquid Engine templates and our [official documentation](https://henriqueslab.github.io/NanoPyx/nanopyx.html).
    - Simple Liquid Engine templates [here](https://github.com/HenriquesLab/NanoPyx/blob/main/src/nanopyx/liquid/_le_template_simple.pyx) and [here](https://github.com/HenriquesLab/NanoPyx/blob/main/src/nanopyx/liquid/_le_template_simple_.py)
    - Fully fledged Liquid Engine templates [here](https://github.com/HenriquesLab/NanoPyx/blob/main/src/nanopyx/liquid/_le_template_advanced.pyx) and [here](https://github.com/HenriquesLab/NanoPyx/blob/main/src/nanopyx/liquid/_le_template_advanced.cl)

## Contributing

Contributions are very welcome.
Please read our [Contribution Guidelines](https://github.com/HenriquesLab/NanoPyx/blob/main/CONTRIBUTING.md) to know how to proceed.

## License

Distributed under the terms of the [CC-By v4.0] license,
"NanoPyx" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[CC-By v4.0]: https://creativecommons.org/licenses/by/4.0/
[file an issue]: https://github.com/HenriquesLab/NanoPyx/issues
[pip]: https://pypi.org/project/pip/

## Development at a glance

## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Structure

[![Structure](https://images.repography.com/33651790/HenriquesLab/NanoPyx/structure/6USKh-PjgkYlbiepDRN9aThOShl3TNx_VkIycH0M6e0/Sqp8CSmE3HObh4_sa8_-IsUByYshpCVQpMuu1E_Fwiw_table.svg)](https://github.com/HenriquesLab/NanoPyx)
