#!/bin/bash -c python3

import os
import shutil
import sys
from inspect import isfunction
import subprocess


def run_cmd(command: str):
    """
    Run a command in the shell
    :param command: command to run
    """
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)


def find_files(root_dir: str, extension: str) -> list:
    """
    Find all files with a given extension in a directory
    :param root_dir: root directory to search
    :param extension: file extension to search for
    :return: list of files
    """
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                target_files.append(os.path.join(root, file))

        # auto remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.listdir(dir_path) == []:
                print("Removing empty directory: ", dir_path)
                os.rmdir(dir_path)
    return target_files


def update_gitignore():
    """
    Update the .gitignore file with common ignores
    """
    try:
        gitignore_lines = open(".gitignore", "r").read().splitlines()
    except FileNotFoundError:
        gitignore_lines = []
    ignores = [
        ".gitignore",
        ".idea",
        ".venv*",
        ".vscode",
        ".pytest_cache",
        "venv",
        "build",
        "dist",
        ".ipynb_checkpoints",
        "__pycache__",
        "*.egg-info",
        "*.so",
        "*.c",
        "src/**/*.html",
        ".coverage*",
        "tests_plots",
        "*.csv",
        "*.npy",
        "*.profile",
        "*.tif",
        ".DS_Store",
        ".coverage",
        "*.pyd",
    ]
    for ignore in ignores:
        if ignore not in gitignore_lines:
            gitignore_lines.append(ignore)

    with open(".gitignore", "w") as f:
        f.write("\n".join(gitignore_lines))


def extract_requirements_from_pyproject():
    """
    Extract the dependencies from pyproject.toml
    Saves into .docker/gha_runners/requirements.txt
    """
    requirements = []
    with open("pyproject.toml", "r") as f:
        txt = f.read()
        # find dependency requirements list
        start = txt.find("dependencies = [") + 15
        end = txt.find("]\n", start) + 1
        requirements += eval(txt[start:end])
        # find test requirements list
        start = txt.find("test = [") + 7
        end = txt.find("]\n", start) + 1
        requirements += eval(txt[start:end])
        # find jupyter requirements list
        start = txt.find("jupyter = [") + 10
        end = txt.find("]\n", start) + 1
        requirements += eval(txt[start:end])
        # find doc requirements list
        start = txt.find("doc = [") + 6
        end = txt.find("]\n", start) + 1
        requirements += eval(txt[start:end])
        # find developer requirements list
        start = txt.find("developer = [") + 12
        end = txt.find("]\n", start) + 1
        requirements += eval(txt[start:end])
    requirements = [line for line in requirements if "nanopyx" not in line]
    with open(os.path.join(".docker", "gha_runners", "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))


def main(mode=None):

    files2clean = " ".join(
        find_files("src", ".so")
        + find_files("src", ".pyc")
        + find_files("src", ".c")
        + find_files("src", ".html")
        + find_files("src", ".profile")
        + find_files("notebooks", ".profile")
        + find_files("src", ".pyd")  # Windows .dll-like file
    )

    python_call = shutil.which("python")
    if sys.platform == "win32":
        remove_call = "del"
    else:
        remove_call = "rm"

    notebook_files = " ".join(find_files("notebooks", ".ipynb"))
    notebook_files += " ".join(find_files("tests", ".ipynb"))
    options = {
        "Build nanopyx extensions": f"{python_call} setup.py build_ext --inplace",
        "Auto-generate pxd files with pyx2pxd": "pyx2pxd src",
        "Clean files": f"{remove_call} {files2clean}"
        if len(files2clean) > 0
        else "echo 'No files to clean'",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Generate .docker/gha_runners/requirements.txt": extract_requirements_from_pyproject,
        "Update .gitignore": update_gitignore,
        "Run pdoc": f"{python_call} -m pdoc src/nanopyx -o docs",
        "Install nanopyx in developer mode": "pip3 install -e '.[all]'",
        "Build nanopyx binary distribution": f"{python_call} setup.py bdist_wheel",
        "Build nanopyx source distribution": f"{python_call} setup.py sdist",
        "Install coding tools": "pip3 install cython-lint",
        "Run cython-lint on pyx files": f"cython-lint {', '.join(find_files('src', '.pyx'))}",
        "Create venv:": "python3 -m venv .venv",
        "Activate venv:": "source .venv/bin/activate",
        "Deactivate venv:": "deactivate",
        "Remove venv:": "rm -rf .venv",
        "Run tests": "pytest",
    }

    # Show the logo
    with open("logo_ascii.txt", "r") as f:
        print(f.read())

    if mode is not None:
        selection = mode

    else:
        # print the options
        print("(⌐⊙_⊙) what do you want to do?")
        for i, option in enumerate(options.keys()):
            cmd = options[option]
            if type(cmd) == str:
                print(
                    f"{i+1}) {option}: [CMD]> {cmd if len(cmd)< 100 else cmd[:100]+'...'}"
                )
            elif isfunction(cmd):
                print(f"{i+1}) {option}: [FUNCTION]> {repr(cmd)}")

        # get the user's selection
        selection = int(input("Enter your selection: ")) - 1

    # print the selected option
    cmd = list(options.values())[selection]
    print(f"- Running command: {repr(cmd)}")
    if type(cmd) == str:
        run_cmd(cmd)
    elif isfunction(cmd):
        cmd()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        modes = sys.argv[1:]
        for mode in modes:
            main(int(mode) - 1)
    else:
        main()
    print("\n(•_•) ( •_•)>⌐■-■\n(⌐■_■) Ready to rock!!")
