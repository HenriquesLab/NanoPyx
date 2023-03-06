import os
import shutil
import sys
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PYTHON_ALL_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
PYTHON_DEFAULT_VERSION = "3.8"
MACOS_INSTALL_DEPENDENCIES = False
if os.environ.get("MACOS_INSTALL_DEPENDENCIES"):
    MACOS_INSTALL_DEPENDENCIES = True

# Platform logic
if sys.platform == "darwin":
    PLATFORM = "macos"
elif sys.platform == "win32":
    PLATFORM = "win"
else:
    PLATFORM = "unix"


# Set nox options
nox.options.reuse_existing_virtualenvs = True

# Some platform specific actions
if PLATFORM == "macos":
    if os.environ.get("NOX_MACOS_INSTALL_DEPENDENCIES"):
        os.system("brew install llvm libomp")


@nox.session(python=PYTHON_ALL_VERSIONS)
def build_wheel(session: nox.Session) -> None:
    """
    Build a wheel
    """
    session.install("build")
    temp_path = session.create_tmp()
    # session.run("python", "-m", "build", "--wheel", "-o", temp_path)
    session.run("pip", "wheel", "--no-deps", "--wheel-dir", temp_path, ".")
    # get the produced wheel name
    wheel_name = [name for name in os.listdir(temp_path) if name.endswith(".whl")][0]

    if PLATFORM == "unix":
        session.install("auditwheel")
        session.run(
            "auditwheel",
            "repair",
            os.path.join(temp_path, wheel_name),
            "-w",
            DIR / "wheelhouse",
        )

    elif PLATFORM == "macos":
        session.install("delocate")
        session.run(
            "delocate-wheel",
            "-v",
            os.path.join(temp_path, wheel_name),
            "-w",
            DIR / "wheelhouse",
        )

    else:
        os.makedirs(DIR / "wheelhouse", exist_ok=True)
        for file in os.listdir(temp_path):
            if file.endswith(".whl"):
                shutil.copy(os.path.join(temp_path, file), DIR / "wheelhouse")


@nox.session(python=PYTHON_DEFAULT_VERSION)
def build_sdist(session: nox.Session) -> None:
    """
    Build an SDist
    """
    session.install("build")
    session.run("python", "-m", "build", "--sdist", "-o", "wheelhouse")


@nox.session(python=PYTHON_ALL_VERSIONS)
def tests_on_source(session):
    """
    Run the test suite
    """
    session.run("pip", "install", "-e", ".[test]")
    session.run("pytest", DIR.joinpath("tests"), "-p", "no:nbmake")
    session.run("coverage", "xml")


@nox.session(python=PYTHON_ALL_VERSIONS)
def tests_on_wheels(session):
    """
    Run the test suite
    """
    python_version_str = f"cp{session.python.replace('.', '')}"
    # find the latest wheel
    wheel_names = [
        wheel
        for wheel in os.listdir("wheelhouse")
        if wheel.endswith(".whl") and python_version_str in wheel
    ]
    wheel_names.sort()
    wheel_name = wheel_names[-1]

    print(python_version_str)
    session.run("pip", "install", "-U", DIR / "wheelhouse" / f"{wheel_name}[test]")
    with session.chdir(".nox"):
        session.run("pytest", DIR.joinpath("tests"))


@nox.session(python=PYTHON_DEFAULT_VERSION)
def generate_docs(session: nox.Session) -> None:
    """
    Generate the docs
    """
    session.run("pip", "install", "-e", ".[doc]")
    session.run("pdoc", DIR / "src" / "nanopyx", "-o", DIR / "docs")
