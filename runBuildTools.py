#!/bin/bash -c python3

import os
import shutil
import subprocess
import sys
from inspect import isfunction


def run_cmd(command: str):
    """
    Run a command in the shell
    :param command: command to run
    """
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)


def get_version():
    # # get the version from the pyproject.toml file
    # with open("pyproject.toml", "r") as f:
    #     txt = f.read()
    #     start = txt.find('version = "') + 11
    #     end = txt.find('"', start)
    #     version = txt[start:end]
    # return version
    import versioneer

    return versioneer.get_version()


def find_files(root_dir: str, extension: str, partner_extension: str = None) -> list:
    """
    Find all files with a given extension in a directory
    :param root_dir: root directory to search
    :param extension: file extension to search for
    :param partner_extension: partner extension to search for (e.g. .pyx and .pxd)
    :return: list of files
    """
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_name = os.path.splitext(file)[0]
            if partner_extension is None:
                if file.endswith(extension):
                    target_files.append(os.path.join(root, file))
            else:
                if file.endswith(extension) and os.path.exists(os.path.join(root, file_name + partner_extension)):
                    target_files.append(os.path.join(root, file))

        # auto remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.listdir(dir_path) == []:
                print("Removing empty directory: ", dir_path)
                os.rmdir(dir_path)
    return target_files


def change_cython_profiler_flag(base_path: str, flag: bool):
    """
    Change the cython profiler flag in all .pyx files
    :param base_path: base path to search for .pyx files
    :param flag: flag to set
    """
    pyx_files = find_files(base_path, ".pyx")
    for pyx_file in pyx_files:
        with open(pyx_file, "r") as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                if line.startswith("# cython:") and f"profile={not flag}" in line:
                    print(f"Changing profile flag to {flag}: {pyx_file}")
                    lines[i] = line.replace(f"profile={not flag}", f"profile={flag}")
                    break
        with open(pyx_file, "w") as f:
            f.write("\n".join(lines))


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
        "docs",
        ".ipynb_checkpoints",
        "__pycache__",
        "*.egg-info",
        "*.so",
        "src/nanopyx/**/*.c",
        "!src/nanopyx/**/*_.c",
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
        "wheelhouse",
        ".nox",
        "build_tools/libs_build",
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
    base_path = os.path.join("src", "nanopyx")
    files2clean = " ".join(
        find_files(base_path, ".so")
        + find_files(base_path, ".pyc")
        + find_files(base_path, ".pyd")
        # + find_files(base_path, ".pyi")
        + find_files(base_path, ".c", partner_extension=".pyx")
        + find_files(base_path, ".html", partner_extension=".pyx")
        + find_files(base_path, ".profile")
        + find_files(os.path.join("tests"), ".profile")
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
        "Auto-generate pxd files via pyx2pxd": f"{python_call} src/scripts/pyx2pxd.py",
        "Auto-copy c to cl files via c2cl": f"{python_call} src/scripts/c2cl.py",
        "Auto-generate code with tag2tag": f"{python_call} src/scripts/tag2tag.py",
        "Clean files": f"{remove_call} {files2clean}" if len(files2clean) > 0 else "echo 'No files to clean'",
        "Run pytest": "pytest -n=auto --nbmake --nbmake-timeout=600",
        "Run nox with all sessions": "pipx run nox",
        "Run nox with build and test wheels": "pipx run nox --session build_wheel build_sdist tests_on_wheels",
        "Run nox with test wheels": "pipx run nox --session tests_on_wheels",
        "Run nox with test source": "pipx run nox --session tests_on_source",
        "Run nox with lint": "pipx run nox --session lint",
        "Run nox with generate docs": "pipx run nox --session generate_docs",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Generate .docker/gha_runners/requirements.txt": extract_requirements_from_pyproject,
        "Update .gitignore": update_gitignore,
        "Run pdoc": f"{python_call} -m pdoc src/nanopyx -o docs",
        "Install nanopyx in developer mode": "pip3 install -e '.[all]'",
        "Run build": "python -m build",
        "Build wheel": "pip wheel '.[all]' -w wheelhouse",
        "Build nanopyx binary distribution": f"{python_call} setup.py bdist_wheel",
        "Build nanopyx source distribution": f"{python_call} setup.py sdist",
        "Install coding tools": "pip3 install cython-lint",
        "Run cython-lint on pyx files": f"cython-lint {', '.join(find_files('src', '.pyx'))}",
        "Create venv:": "python3 -m venv .venv",
        "Activate venv:": "source .venv/bin/activate",
        "Deactivate venv:": "deactivate",
        "Remove venv:": "rm -rf .venv",
        "Test accelerations (requires build first)": "python -c 'import nanopyx.core.utils.mandelbrot_benchmark"
        + " as bench; bench.check_acceleration()'",
    }

    # Show the logo
    with open("logo_ascii.txt", "r") as f:
        print(f.read())

    if mode is not None:
        selection = mode

    else:
        # print the options
        print("Version: ", get_version())
        print("(⌐⊙_⊙) what do you want to do?")
        for i, option in enumerate(options.keys()):
            cmd = options[option]
            if type(cmd) == str:
                print(f"{i+1}) {option}: [CMD]> {cmd if len(cmd)< 100 else cmd[:100]+'...'}")
            elif isfunction(cmd):
                print(f"{i+1}) {option}: [FUNCTION]> {repr(cmd)}")

        # get the user's selection
        selection = int(input("Enter your selection: ")) - 1

    # print the selected option
    cmd = list(options.values())[selection]
    print(
        r'''
          ,~-.
         (  ' )-.          ,~'`-.
      ,~' `  ' ) )       _(   _) )
     ( ( .--.===.--.    (  `    ' )
      `.%%.;::|888.#`.   `-'`~~=~'
      /%%/::::|8888\##\
     |%%/:::::|88888\##|
     |%%|:::::|88888|##|.,-.
     \%%|:::::|88888|##/    )_
      \%\:::::|88888/#/ ( `'  )
       \%\::::|8888/#/(  ,  -'`-.
   ,~-. `%\:::|888/#'(  (     ') )
  (  ) )_ `\__|__/'   `~-~=--~~='
 ( ` ')  ) [VVVVV]
(_(_.~~~'   \|_|/   off we go...
            [XXX]
            `"""'
    '''
    )

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
