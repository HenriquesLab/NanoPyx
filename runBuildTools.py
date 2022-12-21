#!/bin/bash -c python3

import os


def find_files(root_dir, extension):
    pyx_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                pyx_files.append(os.path.join(root, file))
    return pyx_files


def main():

    cython_files = ' '.join(find_files('src', '.so')+find_files('src', '.c')+find_files('src', '.html'))
    notebook_files = ' '.join(find_files('notebooks', '.ipynb'))
    options = {
        "Build nanopyx extensions": "python3 setup.py build_ext --inplace",
        "Build nanopyx binary distribution": "python3 setup.py bdist_wheel",
        "Build nanopyx source distribution": "python3 setup.py sdist",
        "Clear cython files": f"rm {cython_files}",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Install coding tools": "pip install cython-lint",
        "Run cython-lint on pyx files": f"cython-lint {', '.join(find_files('src', '.pyx'))}",
    }

    # print the options
    print("What do you want to do:")
    for i, option in enumerate(options.keys()):
        print(f"{i+1}. {option}: [CMD]> { options[option]}")

    # get the user's selection
    selection = int(input("Enter your selection: ")) - 1

    # print the selected option
    cmd = list(options.values())[selection]
    print(f"- Running command: {repr(cmd)}")
    os.system(cmd)


if __name__ == "__main__":
    main()
