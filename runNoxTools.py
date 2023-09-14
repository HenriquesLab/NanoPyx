#!/bin/bash -c python3

import os
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


def main(mode=None):
    options = {
        "List nox sessions": "nox -l",
        "Run nox with test source": "NPX_PYTEST_ARGS='-n=auto' nox --session test_source",
        "Run nox with build wheels": "nox --session clear_wheelhouse build_wheel build_sdist",
        "Run nox with test wheels": "nox --session test_wheel",
        "Run nox with build and test wheels": "nox --session clear_wheelhouse build_wheel build_sdist test_wheel",
        "Run nox with test on test_pypi":"nox --session test_testpypi",
        "Run nox with test on PyPi":"nox --session test_pypi",
        "Run nox with generate docs": "nox --session generate_docs",
        "Run nox with lint": "nox --session lint",
        # "Run nox with all sessions": "pipx run nox",
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
