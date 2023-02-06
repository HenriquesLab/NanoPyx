import sys
import os


def getVersion():
    # get base path
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"
    )
    # read file
    with open(file_path, "r") as f:
        txt = f.read()
        # find version
        start = txt.find('version = "') + 11
        end = txt.find('"', start)
        version = txt[start:end]
        major, minor, build = version.split(".")
        return major, minor, build, version, txt, file_path


def updateBuildNumber(build_number: str):
    major, minor, build, version, txt, file_path = getVersion()
    # update build number
    new_version = f"{major}.{minor}.{build_number}"
    # update file
    txt = txt.replace(version, new_version)

    # write file
    with open(file_path, "w") as f:
        f.write(txt)


if __name__ == "__main__":
    # get arguments
    args = sys.argv
    if len(args) == 1:
        print(getVersion()[3])
    else:
        # get build number
        build_number = args[1]
        # update version
        updateBuildNumber(build_number)
