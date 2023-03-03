import os


def find_files(root_dir, extension):
    """
    Returns a list of files with given extension in the root directory.
    """
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                path = os.path.join(root, file)
                target_files.append(path)
                # print(f"Found file: {path}")

    return target_files
