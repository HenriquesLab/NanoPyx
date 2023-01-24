#!/bin/bash -c python3

import os
import shutil
import sys
from inspect import isfunction


def find_files(root_dir, extension):
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
    try:
        gitignore_lines = open(".gitignore", "r").read().splitlines()
    except FileNotFoundError:
        gitignore_lines = []
    ignores = [
        ".gitignore",
        ".idea",
        ".venv",
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
        "*.pyd"
    ]
    for ignore in ignores:
        if ignore not in gitignore_lines:
            gitignore_lines.append(ignore)

    with open(".gitignore", "w") as f:
        f.write("\n".join(gitignore_lines))


def main(mode = None):

    clean_files = " ".join(
        find_files("src", ".so")
        + find_files("src", ".pyc")
        + find_files("src", ".c")
        + find_files("src", ".html")
        + find_files("src", ".profile")
        + find_files("notebooks", ".profile")
        + find_files("src", ".pyd") # Windows .dll-like file
    )

    python_call = shutil.which("python")
    if sys.platform == "win32":
        remove_call = 'del'
    else:
        remove_call = 'rm'

    notebook_files = " ".join(find_files("notebooks", ".ipynb"))
    options = {
        "Build nanopyx extensions": f"{python_call} setup.py build_ext --inplace",
        "Auto-generate pxd files with pyx2pxd": f"pyx2pxd src",
        "Clean files": f"{remove_call} {clean_files}"
        if len(clean_files) > 0
        else "echo 'No files to clean'",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Update .gitignore": update_gitignore,
        "Run pdoc": f"{python_call} -m pdoc src/nanopyx -o docs",
        "Install nanopyx in developer mode": "pip3 install -e '.[jupyter,test,doc,developer]'",
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

    print(
        """
  _  _               ___          
 | \| |__ _ _ _  ___| _ \_  ___ __
 | .` / _` | ' \/ _ \  _/ || \ \ /
 |_|\_\__,_|_||_\___/_|  \_, /_\_\\
                         |__/  
     |-- NanoPyx --| Nanoscopy Library...
    """
    )

    if mode is not None:
        selection = mode

    else:
        # print the options
        print("What do you want to do:")
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
        os.system(cmd)
    elif isfunction(cmd):
        cmd()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        modes = sys.argv[1:]
        for mode in modes:
            main(int(mode)-1)
    else:
        main()
