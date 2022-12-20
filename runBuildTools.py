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

    pyx_files = ", ".join(find_files("src", ".pyx"))
    options = {
        "Build nanopyx extensions": "python3 setup.py build_ext --inplace",
        "Clear cython files": f"rm {' '.join(find_files('src', '.so')+find_files('src', '.c')+find_files('src', '.html'))}",
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
