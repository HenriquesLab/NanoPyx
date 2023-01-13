#!/bin/bash -c python3

import os, sys


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
                os.rmdir(dir_path)
    return target_files

def autogenerate_pxd_files(filename):
    """Autogenerate pxd files from pyx files"""
    
    ext = os.path.splitext(filename)[1]
    assert ext == ".pyx", "File must be a pyx file"

    autogen = False
    cdefs = []

    # read the pyx file
    with open(filename, "r") as f:
        pyx_file = f.read()
        lines = pyx_file.splitlines()


        for line in lines:
            if line.startswith("# nanopyx:") and "autogen-pxd=False" in line:
                return
            elif line.startswith("# nanopyx:") and "autogen-pxd=True" in line:
                autogen = True
            elif line.startswith("cdef") and line.endswith(":") and "class" not in line:
                cdefs.append(line[:-1])

    if not autogen:
        return

    # write the pxd file
    pxd_filename = os.path.splitext(filename)[0] + ".pxd"
    with open(pxd_filename, "w") as f:
        print("Autogenerating pxd file: ", pxd_filename)
        f.write("\n".join(cdefs))


def main():

    clean_files = " ".join(
        find_files("src", ".so")
        + find_files("src", ".pyc")
        + find_files("src", ".c")
        + find_files("src", ".html")
        + find_files("src", ".profile")
        + find_files("notebooks", ".profile")
    )

    pyx_files = find_files("src", ".pyx")
    for file in pyx_files:
        autogenerate_pxd_files(file)

    notebook_files = " ".join(find_files("notebooks", ".ipynb"))
    options = {
        "Build nanopyx extensions": "python3 setup.py build_ext --inplace",
        "Clean files": f"rm {clean_files}" if len(clean_files) > 0 else "echo 'No files to clean'",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Install nanopyx in developer mode": "pip3 install -e .",
        "Install nanopyx test packages": "pip install -e .[test]",
        "Build nanopyx binary distribution": "python3 setup.py bdist_wheel",
        "Build nanopyx source distribution": "python3 setup.py sdist",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Install coding tools": "pip install cython-lint",
        "Run cython-lint on pyx files": f"cython-lint {', '.join(find_files('src', '.pyx'))}",
    }

    print("""
                         ,.
                        (_|,.
                       ,' /, )_______   _
                    __j o``-'        `.'-)'
                   ('')     NanoPyx     '
                    `-j                |
                      `-._(           /
         Oink! Oink!     |_\  |--^.  /
        |--- nm ---|    /_]'|_| /_)_/
                            /_]'  /_]'
    """)

    if len(sys.argv) > 1:
        selection = int(sys.argv[1]) - 1

    else:
        # print the options
        print("What do you want to do:")
        for i, option in enumerate(options.keys()):
            cmd = options[option]
            print(
                f"{i+1}) {option}: [CMD]> {cmd if len(cmd)< 100 else cmd[:100]+'...'}"
            )

        # get the user's selection
        selection = int(input("Enter your selection: ")) - 1

    # print the selected option
    cmd = list(options.values())[selection]
    print(f"- Running command: {repr(cmd)}")
    os.system(cmd)


if __name__ == "__main__":
    main()
