import os
from pathlib import Path

try:
    from .__tools__ import find_files
except ImportError:
    from __tools__ import find_files


def get_tags(filename: str, comment_prefix: str = "#"):
    """
    Get all tags from a python or cython file, tags are defined as follows:
    # tag-start: tag_name
    ... code ...
    # tag-end
    :param filename: filename to search for tags
    :return: dictionary of tags and their code
    """
    # example:
    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded
    tag_start_str = comment_prefix + " tag-start: "
    tag_end_str = comment_prefix + " tag-end"
    tags = {}

    with open(filename, "r") as f:
        file_txt = f.read()

        while 1:

            tag_start = file_txt.find(tag_start_str)
            if tag_start == -1:
                return tags
            tag_command_end = file_txt.find("\n", tag_start)
            tag_end = file_txt.find(tag_end_str, tag_command_end)
            if tag_end == -1:
                raise ValueError(f"{filename}: missing {tag_end_str}")

            tag_name = file_txt[
                tag_start + len(tag_start_str) : tag_command_end
            ]
            tag_code = file_txt[tag_command_end + 1 : tag_end]

            tags[tag_name] = tag_code
            print(
                f"Found tag: ...{tag_name.split('.')[-1]} in {os.path.split(filename)[-1]}"
            )
            file_txt = file_txt[tag_end + len(tag_end_str) + 1 :]

    return tags


def replace_tags(filename: str, tags: dict, comment_prefix: str = "#"):
    """
    Replace all tags in a python or cython file, tags are defined as follows:
    # tag-copy: tag_name; replace("old", "new")
    ... code ...
    # tag-end
    :param filename: filename to search for tags
    :param tags: dictionary of tags and their code
    """
    # example:
    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(colsM)", "prange(colsM)") # noqa
    tag_start_str = comment_prefix + " tag-copy: "
    tag_end_str = comment_prefix + " tag-end"
    marker = 0
    file_txt = ""

    with open(filename, "r") as f:
        file_txt = f.read()

    while 1:
        tag_start = file_txt.find(tag_start_str, marker)
        if tag_start == -1:
            break
        tag_command_end = file_txt.find("\n", tag_start)
        tag_end = file_txt.find(tag_end_str, tag_command_end)

        tag_command = file_txt[
            tag_start + len(tag_start_str) : tag_command_end
        ]

        tag_command_elements = tag_command.split(";")
        tag_name = tag_command_elements[0].strip()
        tag_replaces = tag_command_elements[1:]

        if tag_name not in tags:
            raise ValueError(f"{filename}: tag {tag_name} not found")

        tag_code = tags[tag_name]
        for replace_command in tag_replaces:
            if "replace(" not in replace_command:
                continue
            replace_command = replace_command.strip()
            tag_code = eval(f"tag_code.{replace_command}")

        file_txt = (
            file_txt[: tag_command_end + 1] + tag_code + file_txt[tag_end:]
        )
        print(
            f"Adapted code: ...{tag_name.split('.')[-1]} in {os.path.split(filename)[-1]}:{tag_command_end}"
        )
        marker = tag_end

    if marker > 0:
        with open(filename, "w") as f:
            f.write(file_txt)


def parse_files(root_dir: str):
    """
    Parse all files in a directory
    :param root_dir: root directory to search for files
    """

    print(f"Searching {root_dir} for files...")
    py_files = find_files(root_dir, ".py")
    pyx_files = find_files(root_dir, ".pyx")
    cl_files = find_files(root_dir, ".cl")

    print("Auto-generating code for .py and .pyx files...")
    # step 1: get all tags
    tags = {}
    for file in py_files + pyx_files:
        tags.update(get_tags(file, comment_prefix="#"))
    # step 2: replace tags
    for file in py_files + pyx_files:
        replace_tags(file, tags, comment_prefix="#")

    print("Auto-generating code for .cl files...")
    # step 1: get all tags
    tags = {}
    for file in cl_files:
        tags.update(get_tags(file, comment_prefix="//"))
    # step 2: replace tags
    for file in cl_files:
        replace_tags(file, tags, comment_prefix="//")

    # print(tags)


def main():
    print(
        r"""
    ,---------.    ____      .-_'''-.      .`````-. ,---------.    ____      .-_'''-.
    \          \ .'  __ `.  '_( )_   \    /   ,-.  \\          \ .'  __ `.  '_( )_   \
     `--.  ,---'/   '  \  \|(_ o _)|  '  (___/  |   |`--.  ,---'/   '  \  \|(_ o _)|  '
        |   \   |___|  /  |. (_,_)/___|        .'  /    |   \   |___|  /  |. (_,_)/___|
        :_ _:      _.-`   ||  |  .-----.   _.-'_.-'     :_ _:      _.-`   ||  |  .-----.
        (_I_)   .'   _    |'  \  '-   .' _/_  .'        (_I_)   .'   _    |'  \  '-   .'
       (_(=)_)  |  _( )_  | \  `-'`   | ( ' )(__..--.  (_(=)_)  |  _( )_  | \  `-'`   |
        (_I_)   \ (_ o _) /  \        /(_{;}_)      |   (_I_)   \ (_ o _) /  \        /
        '---'    '.(_,_).'    `'-...-'  (_,_)-------'   '---'    '.(_,_).'    `'-...-'
                nanopyx-tag2tag - Wellcome to the meta-programming gardens
    """
    )

    root = Path(__file__).parent.parent.parent
    parse_files(root / "src" / "nanopyx" / "liquid")
    parse_files(root / "tests")


if __name__ == "__main__":
    main()
