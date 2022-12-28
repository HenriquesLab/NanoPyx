"""Build and Install NanoPyx helper file

This module is particularly designed to help build Cython extensions.
Note that any .py containing "# nanopyx-cythonize: True" will be automatically identified as a Pure-Python Cython extension

Example:
    To build cython extensions
        $ python setup.py build_ext --inplace 

Initial creation by:
- Ricardo Henriques, December 2022
Modified by:
- None yet

Todo:
    * Nothing yet
"""

import sys, os, os.path, platform
from Cython.Build import cythonize
from setuptools import Extension, setup

# Some advice when using cython
# consider using the following decorators:
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# @cython.infer_types(True)
# def f(seq):
#  pass

if 'CC' in os.environ:
    print("Using CC={}".format(os.environ['CC']))
else:
    os.environ["CC"] = "gcc"
    print("Using CC={} (set by setup.py)".format(os.environ['CC']))

# Just a basic platform check to engage any platform specific tasks we may need
# Possible values for sys.platform
# ┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━┑
# │ System              │ Value               │
# ┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━┥
# │ Linux               │ linux or linux2 (*) │
# │ Windows             │ win32               │
# │ Windows/Cygwin      │ cygwin              │
# │ Windows/MSYS2       │ msys                │
# │ Mac OS X            │ darwin              │
# │ OS/2                │ os2                 │
# │ OS/2 EMX            │ os2emx              │
# │ RiscOS              │ riscos              │
# │ AtheOS              │ atheos              │
# │ FreeBSD 7           │ freebsd7            │
# │ FreeBSD 8           │ freebsd8            │
# │ FreeBSD N           │ freebsdN            │
# │ OpenBSD 6           │ openbsd6            │
# ┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━┙

if sys.platform == "win32":
    from distutils import msvccompiler
    from platform import architecture

    VC_VERSION = msvccompiler.get_build_version()
    ARCH = "x64" if architecture()[0] == "64bit" else "x86"
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    EXTRA_COMPILE_ARGS = ["/openmp"]
    EXTRA_LING_ARGS = ["/openmp"]

elif sys.platform == "darwin":
    print("Double check you have openmp installed, do 'brew install gcc llvm libomp' if in doubt, you'll also need to install xcode from the app store")
    # brew install gcc llvm libomp
    # sudo xcode-select install
    # export CC='gcc-12'
    # export LDFLAGS="-L/usr/local/opt/libomp/lib"
    # export CPPFLAGS="-I/usr/local/opt/libomp/include"
    INCLUDE_DIRS = ["/usr/local/include"]
    LIBRARY_DIRS = ["/usr/local/lib"]
    pltform = platform.platform().split("-")
    if "macOS" in pltform and "arm" in pltform:
        EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-mcpu=apple-m1", "-Xpreprocessor", "-fopenmp"]
    else:
        EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"]
    EXTRA_LING_ARGS = ["-Xpreprocessor", "-fopenmp"]

elif sys.platform.startswith("linux"):
    INCLUDE_DIRS = ["/usr/local/include", "/usr/include"]
    LIBRARY_DIRS = ["/usr/local/lib", "/usr/lib"]
    EXTRA_COMPILE_ARGS = ["-O3", "-march=native", "-fopenmp"]
    EXTRA_LING_ARGS = ["-fopenmp"]

try:
    import numpy

    INCLUDE_DIRS.append(numpy.get_include())
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def collect_extensions():
    """
    Collect all the directories with Cython extensions and return the list of Extension.
    The function combines static Extension declaration and calls to cythonize to build the list of extensions.
    """
    kwargs = {
        "include_dirs": INCLUDE_DIRS,
        "library_dirs": LIBRARY_DIRS,
        "extra_compile_args": EXTRA_COMPILE_ARGS,
        "extra_link_args": EXTRA_LING_ARGS,
    }

    path = os.path.join("src")

    cython_files = [
        os.path.join(dir, file)
        for (dir, dirs, files) in os.walk(path)
        for file in files
        if file.endswith(".pyx")
        or (
            file.endswith(".py")
            and "# nanopyx-cythonize: True\n" in open(os.path.join(dir, file)).read()
        )
    ]

    cython_extensions = []
    for file in cython_files:
        module = ".".join(os.path.splitext(file)[0].split(os.sep)[1:])
        ext = Extension(module, [file], **kwargs)
        cython_extensions.append(ext)

    print(f"Found following files to build:\n {'; '.join(cython_files)}")

    collected_extensions = cythonize(
        cython_extensions, annotate=True, language_level="3"
    )

    return collected_extensions

setup(
    name="nanopyx",
    ext_modules=collect_extensions(),
    package_data={'': ['**/*.jpg', '**/*.yaml']},
    zip_safe=False,
)
