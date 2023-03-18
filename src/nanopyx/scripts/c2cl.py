import os
from pathlib import Path

from __init__ import find_files


def extract_function_code(file_txt, function_name):
    """
    Extract function code from C file
    :param file_txt: file text
    :param function_name: Function name to extract
    """
    function_code_lines = []

    function_started = False
    brackets = 0

    for line in file_txt.splitlines():
        if function_name in line and "(" in line and ")" in line:
            function_started = True
            brackets = 0

        if function_started:
            brackets += line.count("{")
            brackets -= line.count("}")

            if brackets > 0:
                function_code_lines.append(line)

            if brackets == 0:
                function_code_lines.append(line)
                function_started = False
                break

    if len(function_code_lines) == 0:
        return None

    return "\n".join(function_code_lines)


def copy_c_function_to_cl(cl_filename: str) -> str:
    """
    Copy C functions to CL file
    :param cl_filename: .cl file to copy C functions to
    :return: Message
    """

    ext = os.path.splitext(cl_filename)[1]
    assert ext == ".cl", "File must be a cl file"
    c_filename = os.path.splitext(cl_filename)[0] + ".c"
    if not os.path.exists(c_filename):
        return "- Skipping .cl autocopy (no .c file): " + cl_filename

    # read the cl file
    functions_to_copy = []

    tag_copy_functions = "// c2cl-function: "
    with open(cl_filename, "r") as f:
        cl_txt = f.read()
        if tag_copy_functions not in cl_txt:
            return f"- Skipping .cl autocopy (no {repr(tag_copy_functions)} tag): {cl_filename}"

        cl_lines = cl_txt.splitlines()
        for line in cl_lines:
            if line.startswith(tag_copy_functions):
                functions_to_copy.append(line.replace(tag_copy_functions, ""))

    with open(c_filename, "r") as f:
        c_txt = f.read()

        for function in functions_to_copy:
            tag = tag_copy_functions + function

            c_function_code = extract_function_code(c_txt, function)
            assert (
                c_function_code is not None
            ), f"Could not find function {function} in {c_filename}"

            cl_function_code = extract_function_code(cl_txt, function)
            if cl_function_code is not None:
                # remove previous function
                cl_txt = cl_txt.replace(cl_function_code, "")

            # add new function
            tag_end_position = cl_txt.find(tag) + len(tag) + 1
            cl_txt = (
                cl_txt[:tag_end_position] + c_function_code + cl_txt[tag_end_position:]
            )

    with open(cl_filename, "w") as f:
        f.write(cl_txt)
        return "- Generating .cl file: " + cl_filename


def autocopy_files(root_dir: str):
    print(f"Searching {root_dir} for .cl files...")
    cl_files = find_files(root_dir, ".cl")

    print("Autocopy .cl files...")
    msgs = []
    for file in cl_files:
        msgs.append(copy_c_function_to_cl(file))

    print("\n".join(sorted(msgs)))


def main():
    print(
        r"""
           ______       _
          (_____ \     | |
      ____  ____) )____| |
     / ___)/ ____// ___) |
    ( (___| (____( (___| |
     \____)_______)____)\_)
          nanopyx-c2cl
    """
    )

    path = Path(__file__).parent.parent / "liquid"
    autocopy_files(path)


if __name__ == "__main__":
    main()
