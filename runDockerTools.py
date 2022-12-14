#!/bin/bash -c python3

import os

nb_path = os.path.join(os.getcwd(), "notebooks")

options = {
    "Build nanopyx docker-image": "sudo docker build -t nanopyx .",
    "Run jupyterlab from nanopyx docker-image": f"sudo docker run --name nanopyx1 --rm -p 8888:8888 -v {nb_path}:/notebooks nanopyx",
    "Run bash from from nanopyx docker-image": f"sudo docker run --rm -it nanopyx bash",
    "Start jupyterlab from nanopyx docker-image (needs run first)": f"sudo docker start nanopyx1",
    "Stop jupyterlab from nanopyx docker-image": f"sudo docker stop nanopyx1",
}

# print the options
print("What do you want to do:")
for i, option in enumerate(options.keys()):
    print("{}. {}: [CMD]> {}".format(i+1, option, options[option]))

# get the user's selection
selection = int(input("Enter your selection: "))-1

# print the selected option
cmd = list(options.values())[selection]
print(f"- Running command: {repr(cmd)}")
os.system(cmd)
