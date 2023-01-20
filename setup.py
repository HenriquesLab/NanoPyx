import os
import os.path
import platform
import subprocess
import sys

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup
import setuptools


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
    except:
        return False


def is_homebrew_installed() -> bool:
    try:
        result = subprocess.run(
            ["which", "brew"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except:
        return False


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
    # Lets check if homebrew is installed
    use_openmp_support = True
    print("Checking for openmp support, to run code... ─=≡Σ((( つ◕ل͜◕)つ... blazing fast!!! ")
    if is_xcode_installed():
        print("\t - xcode instalation detected...")
        if is_homebrew_installed():
            print("\t - brew instalation detected...")
            brew_list = run_command("brew list").split()
            packages = ["gcc", "llvm", "libomp"]
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
        gcc_list = run_command("ls /usr/local/bin/").split()
        # Lets try to set the correct gcc version for compilation
        versions = ["12", "11", "10", "9"]
        found = False
        for version in versions:
            gcc_cmd = f"gcc-{version}"
            if gcc_cmd in gcc_list:
                print(f"\t - {gcc_cmd} detected: export CC='{gcc_cmd}'")
                os.environ["CC"] = gcc_cmd
                found = True
        if not found:
            print(
                "GCC not detected on standard directory, looking in brew install folders"
            )
            gcc_list = run_command("ls /opt/homebrew/Cellar/gcc/").split()
            for version in versions:
                for gcc_dir in gcc_list:
                    if version in gcc_dir:
                        gcc_cmd = f"gcc-{version}"
                        print(f"\t - {gcc_cmd} detected: export CC='{gcc_cmd}'")
                        os.environ["CC"] = gcc_cmd

    # brew install gcc llvm libomp
    # sudo xcode-select install
    # export LDFLAGS="-L/usr/local/opt/libomp/lib"
    # export CPPFLAGS="-I/usr/local/opt/libomp/include"
    INCLUDE_DIRS = ["/usr/local/include"]
    LIBRARY_DIRS = ["/usr/local/lib"]
    EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math"]
    EXTRA_LING_ARGS = []
    pltform = platform.platform().split("-")
    # EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-mcpu=apple-m1", "-Xpreprocessor", "-fopenmp"]
    # EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"]
    # EXTRA_LING_ARGS = ["-Xpreprocessor", "-fopenmp"]
    if use_openmp_support:
        EXTRA_COMPILE_ARGS += [
            "-fopenmp"
        ]  # ["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"]
        EXTRA_LING_ARGS += ["-fopenmp"]  # ["-Xpreprocessor", "-fopenmp"]

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

print(
        """
  _  _               ___          
 | \| |__ _ _ _  ___| _ \_  ___ __
 | .` / _` | ' \/ _ \  _/ || \ \ /
 |_|\_\__,_|_||_\___/_|  \_, /_\_\\
                         |__/  
     |-- n🐷 --| 🔬 Library...
 Now compiling... hold on to your 🧢
    """
)

# cython options
# REF: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-options
Options.cimport_from_pyx = False

setup(
    build_ext={"inplace": 1},
    ext_modules=collect_extensions(),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={"": ["**/*.jpg", "**/*.yaml"]},
    zip_safe=False,
)
