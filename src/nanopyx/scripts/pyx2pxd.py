import os
from pathlib import Path

from .__tools__ import find_files


def autogenerate_pxd_file(pyx_filename: str):
    """
    Autogenerate pxd file for given pyx file
    :param pyx_filename: .pyx file to autogenerate pxd file for
    """

    ext = os.path.splitext(pyx_filename)[1]
    assert ext == ".pyx", "File must be a pyx file"

    pxd_filename = os.path.splitext(pyx_filename)[0] + ".pxd"

    autogen = False
    pxd_prefix_txt = ""
    pxd_lines = [
        "# Code below is autogenerated by pyx2pxd",
        "",
    ]

    # extract pre-existing pxd imports
    if os.path.exists(pxd_filename):
        with open(pxd_filename, "r") as f:
            pxd_file_txt = f.read()

            tag = "# pyx2pxd: starting point"
            if tag in pxd_file_txt:
                splitter = pxd_file_txt.find(tag) + len(tag)
                pxd_prefix_txt = pxd_file_txt[:splitter] + "\n"
                pxd_file_txt = pxd_file_txt[splitter:]

            lines = pxd_file_txt.splitlines()
            for line in lines:
                if line.startswith("from"):
                    pxd_lines.append(line)

    # read the pyx file
    with open(pyx_filename, "r") as f:
        pyx_txt = f.read()
        lines = pyx_txt.splitlines()

        for line in lines:
            if line.startswith("# cython:") and "autogen_pxd=False" in line:
                return "- Skipping .pxd autogen (autogen_pxd=False): " + pyx_filename
            elif line.startswith("# cython:") and "autogen_pxd=True" in line:
                autogen = True
            elif line.startswith("cdef class"):
                pxd_lines.append("")
                pxd_lines.append(line)
            elif "# autogen_pxd: " in line:
                pxd_lines.append(line.replace("# autogen_pxd: ", ""))
            elif "# pyx2pxd: " in line:
                pxd_lines.append(line.replace("# pyx2pxd: ", ""))
            elif (
                line.startswith("cdef")
                or line.startswith("    cdef")
                or line.startswith("cpdef")
                or line.startswith("    cpdef")
            ) and ("):" in line or ") nogil:" in line):
                pxd_lines.append(line[: line.rfind(":")])

        pxd_lines.append("")

    if not autogen:
        return "- Skipping .pxd autogen (not enabled): " + pyx_filename

    # write the pxd file
    pxd_text = pxd_prefix_txt + "\n".join(pxd_lines)

    if os.path.exists(pxd_filename) and open(pxd_filename).read() == pxd_text:
        return (
            "- Skipping .pxd autogen (already exists, no changes needed): "
            + pyx_filename
        )

    with open(pxd_filename, "w") as f:
        f.write(pxd_text)
        return "- Generating .pxd file: " + pxd_filename


def autogenerate_pxd_files(root_dir: str):
    """
    Autogenerate pxd files for all .pyx files in the root directory
    :param root_dir: Root directory to search
    """

    print(f"Searching {root_dir} for .pyx files...")
    pyx_files = find_files(root_dir, ".pyx")
    print("Autogenerating .pxd files...")
    msgs = []
    for file in pyx_files:
        msgs.append(autogenerate_pxd_file(file))

    print("\n".join(sorted(msgs)))


def main():
    print(
        r"""
                        ______                 __
    .-----.--.--.--.--.|__    |.-----.--.--.--|  |
    |  _  |  |  |_   _||    __||  _  |_   _|  _  |
    |   __|___  |__.__||______||   __|__.__|_____|
    |__|  |_____|              |__|
                  nanopyx-pyx2pxd
    """
    )

    path = Path(__file__).parent.parent
    autogenerate_pxd_files(path)


if __name__ == "__main__":
    main()
