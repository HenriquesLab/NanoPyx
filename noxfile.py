import nox
from pathlib import Path
import sys
import shutil
import os

DIR = Path(__file__).parent.resolve()
PYTHON_ALL_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
PYTHON_DEFAULT_VERSION = "3.8"
LINT_DEPENDENCIES = [
    "black==22.8.0",
    "flake8==4.0.1",
    "flake8-bugbear==21.11.29",
    "mypy==0.930",
    "types-jinja2",
    "packaging>=20.0",
    "isort==5.10.1",
]

# Platform logic
if sys.platform == "darwin":
    PLATFORM = "macos"
elif sys.platform == "win32":
    PLATFORM = "win"
else:
    PLATFORM = "unix"


# Set nox options
nox.options.sessions = ["tests_on_source", "generate_docs"]
if PLATFORM == "macos":
    # build_docs fail on Windows, even if `chcp.com 65001` is used
    # nox.options.sessions = ["tests", "lint", "build_man"]
    pass
else:
    # nox.options.sessions = ["tests", "lint", "build_docs", "build_man"]
    pass
nox.options.reuse_existing_virtualenvs = True


@nox.session(python=PYTHON_ALL_VERSIONS)
def build_wheel(session: nox.Session) -> None:
    """
    Build a wheel
    """
    session.install("build")
    temp_path = session.create_tmp()
    # session.run("python", "-m", "build", "--wheel", "-o", temp_path)
    session.run("pip", "wheel", "--no-deps", "--wheel-dir", temp_path, ".")
    if PLATFORM == "unix":
        session.install("auditwheel")
        session.run(
            "auditwheel",
            "repair",
            os.path.join(temp_path, "*.whl"),
            "-w",
            DIR / "wheelhouse",
        )
    elif PLATFORM == "macos":
        session.install("delocate")
        session.run(
            "delocate-wheel",
            os.path.join(temp_path, "*.whl"),
            "-w",
            DIR / "wheelhouse",
        )
    else:
        shutil.copy(temp_path, DIR / "wheelhouse")


@nox.session(python=PYTHON_DEFAULT_VERSION)
def build_sdist(session: nox.Session) -> None:
    """
    Build an SDist
    """
    session.install("build")
    session.run("python", "-m", "build", "--sdist", "-o", "wheelhouse")


@nox.session(python=PYTHON_DEFAULT_VERSION)
def lint(session):
    """
    Run the linters
    """
    session.install(*LINT_DEPENDENCIES)
    files = [str(Path("src") / "nanopyx"), "tests"] + [
        str(p) for p in Path(".").glob("*.py")
    ]
    session.run("isort", "--check", "--diff", "--profile", "black", *files)
    session.run("black", "--check", *files)
    session.run("flake8", *files)
    session.run(
        "mypy",
        "--strict-equality",
        "--no-implicit-optional",
        "--warn-unused-ignores",
        *files,
    )


@nox.session(python=PYTHON_ALL_VERSIONS)
def tests_on_source(session):
    """
    Run the test suite
    """
    session.run("pip", "install", "-e", ".[test]")
    with session.chdir(".nox"):
        session.run("pytest", DIR.joinpath("tests"))


@nox.session(python=PYTHON_ALL_VERSIONS)
def tests_on_wheels(session):
    """
    Run the test suite
    """
    session.run("pip", "install", "-U", "nanopyx[test]", "--find-links", "wheelhouse")
    with session.chdir(".nox"):
        session.run("pytest", DIR.joinpath("tests"))


@nox.session(python=PYTHON_DEFAULT_VERSION)
def generate_docs(session: nox.Session) -> None:
    """
    Generate the docs
    """
    session.run("pip", "install", "-e", ".[doc]")
    session.run("pdoc", "src/nanopyx", "-o", DIR / "docs")
