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
                print("Removing empty directory: ", dir_path)
                os.rmdir(dir_path)
    return target_files


def autogenerate_pxd_files(filename):
    """Autogenerate pxd files from pyx files"""

    ext = os.path.splitext(filename)[1]
    assert ext == ".pyx", "File must be a pyx file"

    pxd_filename = os.path.splitext(filename)[0] + ".pxd"

    autogen = False
    cdefs = []

    # extract pre-existing pxd imports
    if os.path.exists(pxd_filename):
        with open(pxd_filename, "r") as f:
            pxd_file = f.read()
            lines = pxd_file.splitlines()
            ignore = False
            for line in lines:
                if "# autogen_pxd - ignore start" in line:
                    ignore = True
                elif "# autogen_pxd - ignore end" in line:
                    ignore = False

                if line.startswith("from") or ignore:
                    cdefs.append(line)

    cdefs.append("")

    # read the pyx file
    with open(filename, "r") as f:
        pyx_file = f.read()
        lines = pyx_file.splitlines()

        for line in lines:
            if line.startswith("# nanopyx:") and "autogen_pxd=False" in line:
                return
            elif line.startswith("# nanopyx:") and "autogen_pxd=True" in line:
                autogen = True
            elif line.startswith("cdef class"):
                cdefs.append("")
                cdefs.append(line)
            elif (
                (line.startswith("cdef") or line.startswith("    cdef"))
                and line.endswith(":")
                and ")" in line
            ):
                cdefs.append(line[:-1])

        cdefs.append("")

    if not autogen:
        return

    # write the pxd file
    pxd_text = "\n".join(cdefs)

    if not os.path.exists(pxd_filename) or open(pxd_filename).read() != pxd_text:
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
        "Clean files": f"rm {clean_files}"
        if len(clean_files) > 0
        else "echo 'No files to clean'",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Run pdoc": "python -m pdoc src/nanopyx -o docs",
        "Install nanopyx in developer mode": "pip3 install -e .",
        "Install nanopyx test packages": "pip install -e .[test]",
        "Build nanopyx binary distribution": "python3 setup.py bdist_wheel",
        "Build nanopyx source distribution": "python3 setup.py sdist",
        "Install coding tools": "pip install cython-lint",
        "Run cython-lint on pyx files": f"cython-lint {', '.join(find_files('src', '.pyx'))}",
    }

    print(
        """
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
    """
    )

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
