import os
import shutil
import sys
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PYTHON_ALL_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
PYTHON_DEFAULT_VERSION = "3.10"

# Platform logic
if sys.platform == "darwin":
    PLATFORM = "macos"
elif sys.platform == "win32":
    PLATFORM = "win"
else:
    PLATFORM = "unix"

# Some platform specific actions
if PLATFORM == "macos":
    if os.environ.get("NPX_MACOS_INSTALL_DEPENDENCIES", False):
        os.system("brew install llvm libomp")


@nox.session(python=PYTHON_ALL_VERSIONS)
def build_wheel(session: nox.Session) -> None:
    """Build a wheel"""
    session.install("build")
    temp_path = session.create_tmp()
    session.run("pip", "wheel", "--no-deps", "--wheel-dir", temp_path, ".")
    wheel_name = [
        name for name in os.listdir(temp_path) if name.endswith(".whl")
    ][0]

    if PLATFORM == "unix" and os.environ.get("NPX_LINUX_FIX_WHEELS", False):
        session.install("auditwheel")
        session.run(
            "auditwheel",
            "repair",
            os.path.join(temp_path, wheel_name),
            "-w",
            DIR / "wheelhouse",
        )
    elif PLATFORM == "macos":
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
    """Build an SDist"""
    session.install("build")
    session.run("python", "-m", "build", "--sdist", "-o", "wheelhouse")


@nox.session(python=PYTHON_DEFAULT_VERSION)
def clear_wheelhouse(session: nox.Session) -> None:
    """Clear the wheelhouse"""
    shutil.rmtree(DIR / "wheelhouse", ignore_errors=True)


def _run_pytest(session, *extra_args):
    """Helper to enforce sequential pytest execution"""
    args = [str(DIR / "tests")]
    if extra_args:
        args.extend(extra_args)
    # 🚫 Force sequential pytest, even if xdist is installed
    session.run("pytest", "-n", "0", *args)
    session.run("coverage", "xml")


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_source(session):
    """Run tests from source"""
    extra_args = os.environ.get("NPX_PYTEST_ARGS", "").split()
    session.run("pip", "install", "-e", ".[test]")
    _run_pytest(session, *extra_args)


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_wheel(session):
    """Run tests from wheel"""
    python_version_str = f"cp{session.python.replace('.', '')}"
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
        extra_args = os.environ.get("NPX_PYTEST_ARGS", "").split()
        _run_pytest(session, *extra_args)


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_testpypi(session):
    """Run tests against TestPyPI"""
    session.run(
        "pip",
        "install",
        "-U",
        "--extra-index-url",
        "https://testpypi.python.org/pypi",
        "nanopyx[all]",
    )
    with session.chdir(".nox"):
        extra_args = os.environ.get("NPX_PYTEST_ARGS", "").split()
        _run_pytest(session, *extra_args)


@nox.session(python=PYTHON_ALL_VERSIONS)
def test_pypi(session):
    """Run tests against PyPI"""
    session.run("pip", "install", "-U", "nanopyx[all]")
    with session.chdir(".nox"):
        extra_args = os.environ.get("NPX_PYTEST_ARGS", "").split()
        _run_pytest(session, *extra_args)


@nox.session(python=PYTHON_DEFAULT_VERSION)
def generate_docs(session: nox.Session) -> None:
    """Generate the docs"""
    session.run("pip", "install", "-e", ".[doc, optional]")
    session.run("pdoc", DIR / "src" / "nanopyx", "-o", DIR / "docs")
