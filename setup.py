import os
import os.path
import platform
import subprocess
import sys

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

import versioneer

# EXTRA_C_FILES_PATH = os.path.join(os.path.split(__file__)[0], "src", "include")
EXTRA_C_FILES_PATH = os.path.join("src", "include")
INCLUDE_DIRS = [EXTRA_C_FILES_PATH]
LIBRARY_DIRS = []
EXTRA_COMPILE_ARGS = []
EXTRA_LING_ARGS = []


def run_command(command: str) -> str:
    result = subprocess.run(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def is_xcode_installed() -> bool:
    try:
        result = subprocess.run(
            ["xcode-select", "--print-path"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_homebrew_installed() -> bool:
    try:
        result = subprocess.run(
            ["which", "brew"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except Exception:
        return False


def get_mpicc_path():
    include = []
    library = []
    try:
        result = subprocess.run(
            ["mpicc", "-showme"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        txt = result.stdout.decode("utf-8").strip()
        for arg in txt.split(" "):
            if arg.startswith("-I"):
                include.append(arg[2:])
            if arg.startswith("-L"):
                library.append(arg[2:])
        return include, library
    except Exception:
        return None


def search_for_c_files_referrenced_in_pyx_text(text: str):
    c_files = []
    for line in text.splitlines():
        # check if we explicitly name a c file to include
        if "# nanopyx-c-file: " in line:
            c_file = line.split("# nanopyx-c-file: ")[1].strip()
            if c_file not in c_files:
                c_files.append(c_file)

        # search for header imports and check if the equivalent c files exist
        elif "cdef extern from" in line:
            c_file = line.split('"')[1].strip()
            c_file = os.path.splitext(c_file)[0] + ".c"
            if c_file not in c_files:
                c_files.append(c_file)

    # search for the file path under EXTRA_C_FILES_PATH path tree
    for i, c_file in enumerate(c_files):
        c_file_path = os.path.join(EXTRA_C_FILES_PATH, c_file)
        if os.path.exists(c_file_path):
            c_files[i] = c_file_path
        else:
            # try to find the file
            for root, dirs, files in os.walk(EXTRA_C_FILES_PATH):
                # Check if the current directory contains the target file
                if c_file in files:
                    # If found, print the full path to the file
                    c_file = os.path.join(root, c_file)
                    c_files[i] = c_file
                    break

    return c_files


if sys.platform == "win32":
    from distutils import msvccompiler
    from platform import architecture

    VC_VERSION = msvccompiler.get_build_version()
    ARCH = "x64" if architecture()[0] == "64bit" else "x86"
    INCLUDE_DIRS += []
    LIBRARY_DIRS += []
    EXTRA_COMPILE_ARGS += ["/openmp"]
    EXTRA_LING_ARGS += ["/openmp"]

elif sys.platform == "darwin":
    INCLUDE_DIRS += ["/usr/local/include"]
    LIBRARY_DIRS += ["/usr/local/lib"]
    EXTRA_COMPILE_ARGS += ["-O3", "-ffast-math"]
    EXTRA_LING_ARGS += []

    # Lets check if homebrew is installed
    use_openmp_support = True
    print(
        "Checking for openmp support, to run code... ─=≡Σ((( つ◕ل͜◕)つ... blazing fast!!! "
    )
    if is_xcode_installed():
        print("\t - xcode instalation detected...")
        if is_homebrew_installed():
            print("\t - brew instalation detected...")
            brew_list = run_command("brew list").split()
            packages = ["llvm", "libomp"]  # , "open-mpi"]  # "gcc"
            for package in packages:
                if package in brew_list:
                    print(f"\t - {package} instalation detected...")
                else:
                    print(
                        f"\t - {package} instalation not detected: consider running 'brew install {' '.join(packages)}'"
                    )
                    use_openmp_support = False
                    break
        else:
            print(
                "\t - brew instalation not detected, consider installing from https://brew.sh/"
            )
            use_openmp_support = False
    else:
        print(
            "\t - xcode instalation not detected, consider installing from the App Store"
        )
        use_openmp_support = False

    if use_openmp_support:
        # some helpful info here REF: https://mac.r-project.org/openmp/
        INCLUDE_DIRS += [
            run_command("brew --prefix libomp").split()[0] + "/include",
            # "/usr/local/opt/llvm/include", - if uncommented breaks cross-compilation
        ]
        LIBRARY_DIRS += [
            run_command("brew --prefix libomp").split()[0] + "/lib",
            # "/usr/local/opt/llvm/lib", - if uncommented breaks cross-compilation
        ]
        # include, library = get_mpicc_path()
        # INCLUDE_DIRS += include
        # LIBRARY_DIRS += library
        EXTRA_COMPILE_ARGS += ["-Xclang", "-fopenmp"]  # , "-lmpi"]
        EXTRA_LING_ARGS += ["-lomp"]  # ["-fopenmp", "-lmpi"]

    pltform = platform.platform().split("-")
    # EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-mcpu=apple-m1", "-Xpreprocessor", "-fopenmp"]
    # EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"]
    # EXTRA_LING_ARGS = ["-Xpreprocessor", "-fopenmp"]


elif sys.platform.startswith("linux"):
    INCLUDE_DIRS += ["/usr/local/include", "/usr/include"]
    LIBRARY_DIRS += ["/usr/local/lib", "/usr/lib"]
    EXTRA_COMPILE_ARGS += ["-O3", "-march=native", "-fopenmp"]
    EXTRA_LING_ARGS += ["-fopenmp"]

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
    extra_c_files = []

    for file in cython_files:
        module = ".".join(os.path.splitext(file)[0].split(os.sep)[1:])
        sources = [file]

        # - Analyse code for extra c files -
        # First search in the pxd file, if it exists
        pxd_file = os.path.join(
            os.path.dirname(file),
            os.path.basename(file).split(".")[0] + ".pxd",
        )
        if os.path.exists(pxd_file):
            with open(pxd_file, "r") as f:
                extra_c_files_candadates = search_for_c_files_referrenced_in_pyx_text(
                    f.read()
                )
            sources += extra_c_files_candadates
            extra_c_files += extra_c_files_candadates

        # Now search in the pyx file
        with open(file, "r") as f:
            extra_c_files_candadates = search_for_c_files_referrenced_in_pyx_text(
                f.read()
            )
            sources += extra_c_files_candadates
            extra_c_files += extra_c_files_candadates

        # Remove redundancy
        sources = list(set(sources))
        extra_c_files = list(set(extra_c_files))

        # Remove files that don't exist
        sources = [file for file in sources if os.path.exists(file)]
        extra_c_files = [file for file in extra_c_files if os.path.exists(file)]

        # Make sure we have all the include paths
        for path in extra_c_files:
            if os.path.split(path)[0] not in INCLUDE_DIRS:
                INCLUDE_DIRS.append(os.path.split(path)[0])

        ext = Extension(module, sources, **kwargs)
        cython_extensions.append(ext)

    print(f"Found following .pyx files to build:\n {'; '.join(cython_files)}")
    print(f"Found the extra c files to build:\n {'; '.join(extra_c_files)}")

    collected_extensions = cythonize(
        cython_extensions, annotate=True, language_level="3"
    )

    return collected_extensions


VERSION = versioneer.get_version()
# Show the logo
print(
    r"""
  _  _               ___
 | \| |__ _ _ _  ___| _ \_  ___ __
 | .` / _` | ' \/ _ \  _/ || \ \ /
 |_|\_\__,_|_||_\___/_|  \_, /_\_\
      [ (( X )) ]     *|<|__/==/==
  |-- Python Nanoscopy Library --|
"""
    + "  "
    + f"v{VERSION}".center(len("|-- Python Nanoscopy Library --|"))
)

# cython options
# REF: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-options
Options.cimport_from_pyx = False

setup(
    build_ext={"inplace": 1},
    ext_modules=collect_extensions(),
    zip_safe=False,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
