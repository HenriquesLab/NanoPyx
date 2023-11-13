import os
from pathlib import Path

def extract_batch_code(list_of_cfunction):

    headers = []
    functions = []

    for cf in list_of_cfunction:
        h,f = extract_function_code(*cf)

        headers.append(h)
        functions.append(f)
    return headers, functions

def extract_function_code(file_txt, function_name):
    """
    Extract function code from C file
    :param file_txt: file path
    :param function_name: Function name to extract
    :return: Function signature and code (str, str)
    """

    file_txt = open(confirm_c_file_absolute_path(file_txt)).read()

    # make all "float *" global
    file_txt = file_txt.replace("float *", "__global float *")
    file_txt = file_txt.replace("float* ", "__global float *")

    function_code_lines = []

    function_started = False
    brackets = 0

    for line in file_txt.splitlines():
        if (
            function_name in line
            and "(" in line
            and ")" in line
            and ";" not in line
        ):
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

    function_signature = function_code_lines[0].split("{")[0].strip() + ";"
    # print("\n".join(function_code_lines))
    return function_signature, "\n".join(function_code_lines)


def find_function_header(file_txt, function_name):
    """
    Find function header in C file
    :param file_txt: file text
    :param function_name: Function name to extract
    :return: Function header (str)
    """
    p_function_name = file_txt.find(function_name)
    if p_function_name == -1:
        return None

    p_next_semicolon = file_txt.find(";", p_function_name)
    if p_next_semicolon == -1:
        return None

    p_next_open_bracket = file_txt.find("{", p_function_name)
    if p_next_open_bracket != -1 and p_next_open_bracket < p_next_semicolon:
        return None

    header = file_txt[p_function_name : p_next_semicolon + 1]

    # add previous word to header
    header = file_txt[:p_function_name].split()[-1] + " " + header
    print(header)

    return header


def extract_define_code(file_txt, define_name):
    """
    Extract define code from C file
    :param file_txt: file text
    :param define_name: Define name to extract
    :return: Define code (str)
    """
    for line in file_txt.splitlines():
        if define_name in line and line.strip().startswith("#define"):
            return line


def confirm_c_file_absolute_path(c_filename):
    """
    Confirm that the C file is in the include directory
    :param c_filename: C file name
    :return: absolute path of C file
    """
    if os.path.exists(c_filename):
        return os.path.abspath(c_filename)

    # find file in include directory
    file_path = Path(__file__).parent.parent / "include" / c_filename
    if os.path.exists(file_path):
        return os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file {file_path}")

    return None

if __name__=="__main__":

    inter_name = 'nearest_neighbor'
    c_function_names = [(f'_c_interpolation_{inter_name}.c','_c_interpolate')]

    c_functions = [extract_function_code(*c) for c in c_function_names]

    print(c_functions)