#!/bin/bash -c python3

import os

nb_path = os.path.join(os.getcwd(), "notebooks")

options = {
    "Build nanopyx docker-image": "sudo docker build -t nanopyx .",
    "Run jupyterlab from nanopyx docker-image": f"sudo docker run --rm --name nanopyx1 -p 8888:8888 -v {nb_path}:/notebooks nanopyx",  # potentially add --rm
    "Run bash from from nanopyx docker-image": "sudo docker run -it nanopyx bash",  # potentially add --rm
    "Start jupyterlab from nanopyx docker-image (needs run first)": "sudo docker start nanopyx1",
    "Stop jupyterlab from nanopyx docker-image": "sudo docker stop nanopyx1",
    "Remove nanopyx docker-image": "sudo docker rm nanopyx1 || sudo docker image rm nanopyx || sudo docker image prune -f || docker container prune -f || sudo docker system df",
}

# print the options
print("What do you want to do:")
for i, option in enumerate(options.keys()):
    print("{}. {}: [CMD]> {}".format(i + 1, option, options[option]))

# get the user's selection
selection = int(input("Enter your selection: ")) - 1

# print the selected option
cmd = list(options.values())[selection]
print(f"- Running command: {repr(cmd)}")
os.system(cmd)
