import os
import shutil
import sys
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PYTHON_ALL_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]
PYTHON_DEFAULT_VERSION = "3.10"

# Platform logic
if sys.platform == "darwin":
    PLATFORM = "macos"
elif sys.platform == "win32":
    PLATFORM = "win"
else:
    PLATFORM = "unix"


# Set nox options
nox.options.reuse_existing_virtualenvs = False

# Some platform specific actions
if PLATFORM == "macos":
    if os.environ.get("NPX_MACOS_INSTALL_DEPENDENCIES", False):
        os.system("brew install llvm libomp")


@nox.session(python=PYTHON_ALL_VERSIONS)
def build_wheel(session: nox.Session) -> None:
    """
    Build a wheel for ARM64 on macOS or AMD64 on Linux/Windows
    """
    import platform

    arch = platform.machine()
    is_macos = sys.platform == "darwin"
    is_linux = sys.platform.startswith("linux")
    is_windows = sys.platform == "win32"

    # Skip if not on the right arch/platform combination
    if is_macos and arch != "arm64":
        session.skip(
            f"Skipping macOS build on non-arm64 architecture (found {arch})"
        )
    elif (is_linux or is_windows) and arch != "x86_64" and arch != "AMD64":
        session.skip(
            f"Skipping {PLATFORM} build on non-x86_64 architecture (found {arch})"
        )

    session.install("build")
    temp_path = session.create_tmp()
    session.run("pip", "wheel", "--no-deps", "--wheel-dir", temp_path, ".")

    # Get wheel name
    wheel_name = [
        name for name in os.listdir(temp_path) if name.endswith(".whl")
    ][0]

    if is_linux and os.environ.get("NPX_LINUX_FIX_WHEELS", False):
        session.install("auditwheel")
        session.run(
            "auditwheel",
            "repair",
            os.path.join(temp_path, wheel_name),
            "-w",
            DIR / "wheelhouse",
        )
    elif is_macos:
        session.install("delocate==0.10.4")
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


@nox.session(python=PYTHON_DEFAULT_VERSION)
def clear_wheelhouse(session: nox.Session) -> None:
    """
    Clear the wheelhouse
    """
    shutil.rmtree(DIR / "wheelhouse", ignore_errors=True)


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_source(session):
    """
    Run the test suite by directly calling pip install -e .[test] and then pytest
    """
    extra_args = os.environ.get("NPX_PYTEST_ARGS", "")
    session.run("pip", "install", "-e", ".[test]")
    if extra_args != "":
        extra_args = extra_args.split(" ")
        session.run("pytest", DIR.joinpath("tests"), *extra_args)
    else:
        session.run("pytest", DIR.joinpath("tests"))
    session.run("coverage", "xml")


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_wheel(session):
    """
    Run the test suite by installing the wheel, changing directory and then calling pytest
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

    session.run(
        "pip", "install", "-U", DIR / "wheelhouse" / f"{wheel_name}[test]"
    )
    with session.chdir(".nox"):
        extra_args = os.environ.get("NPX_PYTEST_ARGS", "")
        if extra_args != "":
            extra_args = extra_args.split(" ")
            session.run("pytest", DIR.joinpath("tests"), *extra_args)
        else:
            session.run("pytest", DIR.joinpath("tests"))


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_testpypi(session):
    session.run(
        "pip",
        "install",
        "-U",
        "--extra-index-url",
        "https://testpypi.python.org/pypi",
        "nanopyx[all]",
    )
    with session.chdir(".nox"):
        extra_args = os.environ.get("NPX_PYTEST_ARGS", "")
        if extra_args != "":
            extra_args = extra_args.split(" ")
            session.run("pytest", DIR.joinpath("tests"), *extra_args)
        else:
            session.run("pytest", DIR.joinpath("tests"))


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_pypi(session):
    session.run("pip", "install", "-U", "nanopyx[all]")
    with session.chdir(".nox"):
        extra_args = os.environ.get("NPX_PYTEST_ARGS", "")
        if extra_args != "":
            extra_args = extra_args.split(" ")
            session.run("pytest", DIR.joinpath("tests"), *extra_args)
        else:
            session.run("pytest", DIR.joinpath("tests"))


@nox.session(python=PYTHON_DEFAULT_VERSION)
def generate_docs(session: nox.Session) -> None:
    """
    Generate the docs
    """
    session.run("pip", "install", "-e", ".[doc, optional]")
    session.run("pdoc", DIR / "src" / "nanopyx", "-o", DIR / "docs")
